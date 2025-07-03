from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.config import DOCUMENTS_PATH
from src.data_loader import get_supported_extensions, load_documents_from_directory
from src.database import ChatDatabase
from src.models import llm
from src.rag import RAG


class ChatBot:
    """Класс чат-бота."""

    def __init__(self) -> None:
        """Инициализация чат-бота."""

        self.db = ChatDatabase()
        self.use_rag = False
        self.rag = RAG()
        self.documents_path = DOCUMENTS_PATH

    def get_conversations(self) -> list[dict[str, str]]:
        """Получить список диалогов."""

        return self.db.get_conversations()

    def create_new_conversation(self) -> str:
        """Создать новый диалог."""

        return self.db.create_new_conversation()

    def delete_conversation(self, conversation_id: str) -> bool:
        """Удалить диалог."""

        return self.db.delete_conversation(conversation_id)

    def conversation_exists(self, conversation_id: str) -> bool:
        """Проверить, существует ли диалог."""

        return self.db.conversation_exists(conversation_id)

    def get_conversation_history(self, conversation_id: str) -> list[HumanMessage | AIMessage | SystemMessage]:
        """Получить историю диалога."""

        return self.db.get_conversation_history(conversation_id)

    def toggle_rag(self) -> bool:
        """Переключить RAG."""

        self.use_rag = not self.use_rag
        return self.use_rag

    def toggle_more_diverse(self) -> bool:
        """Переключить режим разнообразных ответов."""

        self.rag.change_type_of_search(True)
        return self.rag.more_diverse

    def toggle_more_accurate(self) -> bool:
        """Переключить режим точных ответов."""

        self.rag.change_type_of_search(False)
        return self.rag.more_diverse

    def update_documents(self) -> None:
        """Обновить документы для RAG."""

        documents = load_documents_from_directory(self.documents_path)
        self.rag.add_documents(documents)

    def get_supported_extensions(self) -> list[str]:
        """Получить список поддерживаемых расширений."""

        return get_supported_extensions()

    def chat(self, query: str, conversation_id: str) -> str:
        """Общение с ботом."""

        chat_history = self.get_conversation_history(conversation_id)
        chat_history = trim_messages(
            chat_history,
            max_tokens=150_000,
            strategy="last",
            token_counter=count_tokens_approximately,
            start_on="human",
            include_system=True,
            allow_partial=False,
        )

        if self.use_rag:
            response = self.rag.generate(query, chat_history)
        else:
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "Вы полезный помощник. Будьте краткими и лаконичными."),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{query}"),
                ]
            )

            chain = prompt | llm | StrOutputParser()
            response = chain.invoke({"query": query, "chat_history": chat_history})

        self.db.save_message(conversation_id, "human", query)
        self.db.save_message(conversation_id, "ai", response)

        return response
