import os

from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")  # Корневая директория
# Ожидаемая структура проекта:
# src/
#   config.py
#   ...
# .env
# ...

if os.path.exists(env_path):
    load_dotenv(env_path)
    LLM_NAME = os.getenv("LLM_NAME")
    BASE_URL = os.getenv("BASE_URL")
    API_KEY = os.getenv("API_KEY")
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
    RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME")

    DOCUMENTS_PATH = "data/documents"
    DB_PATH = "data/chat_history/chat_history.db"
    VECTOR_STORE_PATH = "data/vector_store"
else:
    raise FileNotFoundError("Файл .env не найден. Пожалуйста, создайте его.")
