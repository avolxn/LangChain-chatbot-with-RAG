import faiss
from langchain.indexes import SQLRecordManager, index
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.config import DOCUMENTS_PATH, VECTOR_STORE_PATH
from src.data_loader import split_documents
from src.models import embedder, llm, reranker


class RAG:
    """Система Retrieval Augmented Generation с продвинутыми возможностями."""

    def __init__(
        self,
        k: int = 3,
        fetch_k: int = 10,
    ) -> None:
        """
        Инициализация RAG системы.

        Args:
            k: Количество документов для извлечения из векторного хранилища.
            fetch_k: Количество документов для использования после реранжирования.
        """

        self.documents_paths = DOCUMENTS_PATH
        self.vector_store_path = VECTOR_STORE_PATH

        self.k = k
        self.fetch_k = fetch_k
        self.more_diverse = True

        self.record_manager = None
        self.vector_store = None

        try:
            self._init_record_manager()
            self._init_vector_store()
            self._create_retriever()
            self._create_contextualize_query_chain()
            self._create_qa_chain()
        except Exception as e:
            raise RuntimeError(f"Ошибка инициализации RAG: {e}") from e

    def _init_record_manager(self) -> None:
        """
        Инициализация менеджера записей.
        """

        self.record_manager = SQLRecordManager(
            namespace="faiss_index",
            db_url=f"sqlite:///{self.vector_store_path}/record_manager.db",
        )

        try:
            self.record_manager.list_keys()
        except Exception:
            self.record_manager.create_schema()

    def _init_vector_store(self) -> None:
        """
        Загрузить существующее векторное хранилище или создать новое пустое.
        """

        try:
            self.vector_store = FAISS.load_local(
                self.vector_store_path,
                embedder,
                allow_dangerous_deserialization=True,
            )
        except Exception:
            dummy_vector = embedder.embed_query("__dummy__")
            dimension = len(dummy_vector)
            index = faiss.IndexFlatIP(dimension)

            self.vector_store = FAISS(
                embedding_function=embedder,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )
            self.vector_store.save_local(self.vector_store_path)

    def _create_retriever(self) -> None:
        """
        Настроить ретривер с фильтрацией по схожести.
        """

        if self.more_diverse:
            self.retriever = self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": self.k, "fetch_k": self.fetch_k},
            )
        else:
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.fetch_k},
            )

    def _create_contextualize_query_chain(self) -> None:
        """
        Построить цепочку для переформулирования запросов с использованием истории чата.
        """

        contextualize_query_prompt = """
        Учитывая историю чата и последний вопрос пользователя, который может ссылаться на контекст в истории чата,
        сформулируйте отдельно стоящий вопрос, который можно понять без истории чата.
        НЕ отвечайте на вопрос, просто переформулируйте его при необходимости, иначе верните как есть.
        """
        contextualize_query_chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", contextualize_query_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{query}"),
                ]
            )
            | llm
            | StrOutputParser()
        )
        self.contextualize_query_chain = contextualize_query_chain

    def _format_docs(self, docs: list[Document]) -> str:
        """
        Форматировать документы для QA цепочки.

        Args:
            docs: Список документов.

        Returns:
            Отформатированные документы в виде одной строки.
        """

        return "\n\n".join(doc.page_content for doc in docs)

    def _create_qa_chain(self) -> None:
        """
        Построить QA цепочку с продвинутыми правилами промптинга.
        """

        qa_prompt = """
        Вы полезный помощник, который отвечает на вопросы пользователей.

        Используйте следующую информацию из различных источников для ответа на вопрос:
        {context}

        Следуйте этим правилам:
        1. Если предоставленная информация отвечает на вопрос пользователя, используйте ее для точного и полного ответа.
        2. Если информации недостаточно или она не полностью отвечает на вопрос, используйте ее в сочетании с вашими знаниями.
        3. Если информация не относится к вопросу, полагайтесь только на свои знания.
        4. Если вы не уверены или не знаете ответа, честно признайтесь в этом.
        5. Не придумывайте факты или информацию, которой нет в предоставленных документах или ваших знаниях.
        6. Ваш ответ должен быть логически связным.

        Важно: Не упоминайте в своем ответе, что вы используете предоставленные документы или контекст.
        """
        qa_chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", qa_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{query}"),
                ]
            )
            | llm
            | StrOutputParser()
        )
        self.qa_chain = qa_chain

    def change_type_of_search(self, more_diverse: bool) -> None:
        """
        Изменить тип поиска.

        Args:
            more_diverse: Использовать ли более подход, покрывающий больше контекста, для ответа на вопрос.
        """
        self.more_diverse = more_diverse
        self._create_retriever()

    def generate(self, query: str, chat_history: list[SystemMessage | HumanMessage | AIMessage]) -> str:
        """
        Сгенерировать ответ через RAG пайплайн.

        Args:
            query: Запрос пользователя.
            chat_history: История чата.

        Returns:
            Сгенерированный ответ.
        """
        contextualized = self.contextualize_query_chain.invoke({"query": query, "chat_history": chat_history})

        docs = self.retriever.invoke(contextualized)

        if not self.more_diverse:
            docs = reranker.compress_documents(docs, contextualized)

        formatted_docs = self._format_docs(docs)

        answer = self.qa_chain.invoke({"query": query, "chat_history": chat_history, "context": formatted_docs})
        return answer

    def add_documents(self, documents: list[Document]) -> None:
        """
        Добавить новые документы в векторное хранилище.

        Args:
            documents: Список документов для добавления.
        """

        if not documents:
            print("Нет документов для добавления. Поместите их в папку data/documents.")
            return

        splitted_docs = split_documents(documents)

        result = index(
            splitted_docs,
            self.record_manager,
            self.vector_store,
            cleanup="full",
            source_id_key="source",
        )

        self.vector_store.save_local(self.vector_store_path)

        print(f"Результат: {result}")
