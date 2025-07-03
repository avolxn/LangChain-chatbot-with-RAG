from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from src.config import API_KEY, BASE_URL, EMBEDDING_MODEL_NAME, LLM_NAME, RERANKER_MODEL_NAME


def get_llm(temperature: float = 0.3) -> ChatOpenAI:
    """
    Получить языковую модель, совместимую с OpenAI API.

    Args:
        temperature: Температура сэмплирования (выше означает более креативно).

    Returns:
        Языковая модель, совместимая с OpenAI API.
    """
    try:
        llm = ChatOpenAI(
            model=LLM_NAME,
            api_key=API_KEY,
            base_url=BASE_URL,
            temperature=temperature,
        )
        return llm
    except Exception as e:
        raise RuntimeError(f"Ошибка инициализации языковой модели: {e}") from e


def get_embedder() -> HuggingFaceEmbeddings:
    """
    Получить модель эмбеддингов HuggingFace.

    Returns:
        Модель эмбеддингов HuggingFace.
    """

    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        return embeddings
    except Exception as e:
        raise RuntimeError(f"Ошибка инициализации модели эмбеддингов: {e}") from e


def get_reranker(top_n: int = 5) -> CrossEncoderReranker:
    """
    Получить модель реранжирования.

    Args:
        top_n: Количество документов для возврата после реранжирования.

    Returns:
        Модель реранжирования.
    """

    try:
        cross_encoder = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL_NAME)
    except Exception as e:
        raise RuntimeError(f"Ошибка загрузки кросс-энкодера: {e}") from e

    try:
        reranker = CrossEncoderReranker(model=cross_encoder, top_n=top_n)
        return reranker
    except Exception as e:
        raise RuntimeError(f"Ошибка создания реранжера: {e}") from e


# Синглтон
llm = get_llm()
embedder = get_embedder()
reranker = get_reranker()
