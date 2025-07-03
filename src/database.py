import json
import os
import sqlite3
import uuid
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.config import DB_PATH


class ChatDatabase:
    """Класс для операций с базой данных истории чатов."""

    def __init__(self):
        """Инициализация базы данных чатов."""

        self.db_path = DB_PATH
        self._init_db()

    def _init_db(self) -> None:
        """
        Инициализация SQLite базы данных для истории чатов.
        """

        try:
            db_dir = os.path.dirname(self.db_path)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    metadata TEXT
                )
                """)

                cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    conversation_id TEXT,
                    role TEXT,
                    content TEXT,
                    created_at TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
                )
                """)

                conn.commit()
        except Exception as e:
            raise RuntimeError(f"Ошибка инициализации базы данных: {e}") from e

    def get_conversation_history(self, conversation_id: str) -> list[HumanMessage | AIMessage | SystemMessage]:
        """
        Получить историю диалога из базы данных.

        Args:
            conversation_id: ID диалога

        Returns:
            Список объектов сообщений
        """

        messages = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY created_at",
                    (conversation_id,),
                )

                for role, content in cursor.fetchall():
                    if role == "human":
                        messages.append(HumanMessage(content=content))
                    elif role == "ai":
                        messages.append(AIMessage(content=content))
                    elif role == "system":
                        messages.append(SystemMessage(content=content))

                return messages
        except Exception as e:
            raise RuntimeError(f"Ошибка получения истории диалога: {e}") from e

    def save_message(self, conversation_id: str, role: str, content: str) -> None:
        """
        Сохранить сообщение в базу данных.

        Args:
            conversation_id: ID диалога
            role: Роль отправителя сообщения (human, ai, system)
            content: Содержимое сообщения
        """

        try:
            if role not in ["human", "ai", "system"]:
                raise ValueError(f"Неверная роль: {role}. Допустимые роли: human, ai, system.")

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("SELECT 1 FROM conversations WHERE conversation_id = ?", (conversation_id,))
                now = datetime.now().isoformat()

                if not cursor.fetchone():
                    cursor.execute(
                        "INSERT INTO conversations VALUES (?, ?, ?, ?)", (conversation_id, now, now, json.dumps({}))
                    )
                else:
                    cursor.execute(
                        "UPDATE conversations SET updated_at = ? WHERE conversation_id = ?",
                        (now, conversation_id),
                    )

                message_id = str(uuid.uuid4())
                cursor.execute(
                    "INSERT INTO messages VALUES (?, ?, ?, ?, ?)",
                    (message_id, conversation_id, role, content, now),
                )

                conn.commit()
        except Exception as e:
            raise RuntimeError(f"Ошибка сохранения сообщения: {e}") from e

    def get_conversations(self) -> list[dict[str, str]]:
        """
        Получить список всех диалогов.

        Returns:
            Список объектов диалогов
        """

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    "SELECT conversation_id, created_at, updated_at FROM conversations ORDER BY updated_at DESC"
                )

                conversations = []
                for conversation_id, created_at, updated_at in cursor.fetchall():
                    conversations.append({"id": conversation_id, "created_at": created_at, "updated_at": updated_at})

                return conversations
        except Exception as e:
            raise RuntimeError(f"Ошибка получения списка диалогов: {e}") from e

    def create_new_conversation(self) -> str:
        """
        Создать новый диалог и вернуть его ID.

        Returns:
            ID диалога
        """

        try:
            conversation_id = str(uuid.uuid4())
            now = datetime.now().isoformat()

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    "INSERT INTO conversations VALUES (?, ?, ?, ?)", (conversation_id, now, now, json.dumps({}))
                )

                conn.commit()
                return conversation_id
        except Exception as e:
            raise RuntimeError(f"Ошибка создания нового диалога: {e}") from e

    def conversation_exists(self, conversation_id: str) -> bool:
        """
        Проверить, существует ли диалог.

        Args:
            conversation_id: ID диалога для проверки

        Returns:
            True если диалог существует, иначе False
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1 FROM conversations WHERE conversation_id = ?", (conversation_id,))
                return cursor.fetchone() is not None
        except Exception as e:
            raise RuntimeError(f"Ошибка проверки существования диалога: {e}") from e

    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Удалить диалог и все связанные сообщения.

        Args:
            conversation_id: ID диалога для удаления

        Returns:
            True если удален успешно, False если диалог не найден
        """

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute("SELECT 1 FROM conversations WHERE conversation_id = ?", (conversation_id,))
                if not cursor.fetchone():
                    return False

                cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
                cursor.execute("DELETE FROM conversations WHERE conversation_id = ?", (conversation_id,))

                conn.commit()
                return True
        except Exception as e:
            raise RuntimeError(f"Ошибка удаления диалога: {e}") from e
