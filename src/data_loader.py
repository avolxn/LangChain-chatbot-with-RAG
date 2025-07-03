import os

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, UnstructuredHTMLLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from src.config import DOCUMENTS_PATH

LOADER_CONFIGS = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".doc": Docx2txtLoader,
    ".docx": Docx2txtLoader,
    ".html": UnstructuredHTMLLoader,
    ".md": UnstructuredMarkdownLoader,
    # ".json": JSONLoader,
}


def load_documents_from_directory(directory_path: str = DOCUMENTS_PATH) -> list[Document]:
    """
    Загрузить документы из директории.

    Args:
        directory_path: Путь к директории с документами.

    Returns:
        Список документов.
    """

    if not os.path.exists(directory_path):
        raise ValueError(f"Директория {directory_path} не существует")

    if not os.path.isdir(directory_path):
        raise ValueError(f"Путь {directory_path} не является директорией")

    documents = []

    for file_name in tqdm(os.listdir(directory_path), desc="Загрузка документов из директории"):
        file_path = os.path.join(directory_path, file_name)
        if os.path.isfile(file_path):
            _, file_extension = os.path.splitext(file_name)
            loader_class = LOADER_CONFIGS.get(file_extension.lower())
            try:
                loader = loader_class(file_path)
                loaded_docs = loader.load()
                documents.extend(loaded_docs)
            except Exception:  # noqa: S112
                continue

    return documents


def split_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """
    Разделить документы на чанки.

    Args:
        documents: Список документов.
        chunk_size: Размер чанка.
        chunk_overlap: Перекрытие чанков.

    Returns:
        Список разделенных документов.
    """

    if not documents:
        return []

    separators = [
        "\n\n",
        "\n",
        ".",
        "!",
        "?",
        ",",
        " ",
        "",
    ]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        keep_separator=True,
        strip_whitespace=True,
    )

    splitted_documents = text_splitter.split_documents(documents)
    return splitted_documents


def get_supported_extensions() -> list[str]:
    """
    Получить список поддерживаемых расширений файлов.

    Returns:
        Список поддерживаемых расширений.
    """

    return list(LOADER_CONFIGS.keys())
