import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
print("Загрузка...\n")

from src.chat_bot import ChatBot
from src.config import DB_PATH, DOCUMENTS_PATH, VECTOR_STORE_PATH


def setup_directories():
    """Создание необходимых директорий для приложения."""
    os.makedirs(DOCUMENTS_PATH, exist_ok=True)
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)


def main():
    setup_directories()

    chat_bot = ChatBot()

    conversation_id = chat_bot.create_new_conversation()

    print(
        "🤖 Добро пожаловать в чат-бот! Введите '/exit' для завершения работы. Для использования RAG поместите документы в папку data/documents."
    )
    print("\n📋 Доступные команды:")
    print("  /help - показать доступные команды ℹ️")
    print("  /rag - переключить режим RAG 🔄")
    print("  /more_diverse - переключить режим разнообразных ответов 🎨")
    print("  /more_accurate - переключить режим точных ответов 🎯")
    print("  /update_documents - обновить документы для RAG 📚")
    print("  /supported_extensions - показать поддерживаемые расширения 📄")
    print("  /new - создать новый диалог ➕")
    print("  /list - показать список диалогов 📝")
    print("  /select <id> - выбрать диалог по ID 🔍")
    print("  /delete <id> - удалить диалог по ID 🗑️")

    while True:
        query = input("\n👤 Вы: ")

        if not chat_bot.conversation_exists(conversation_id):
            print("⚠️ Текущий диалог не существует. Создание нового...")
            conversation_id = chat_bot.create_new_conversation()
            print(f"✅ Новый диалог создан с ID: {conversation_id}")

        if query.lower() == "/exit":
            break

        if query.strip().startswith("/"):
            command = query.strip()[1:].lower()

            if command == "help":
                print("\n📋 Доступные команды:")
                print("  /help - показать доступные команды ℹ️")
                print("  /rag - переключить режим RAG 🔄")
                print("  /more_diverse - переключить режим разнообразных ответов 🎨")
                print("  /more_accurate - переключить режим точных ответов 🎯")
                print("  /update_documents - обновить документы для RAG 📚")
                print("  /supported_extensions - показать поддерживаемые расширения 📄")
                print("  /new - создать новый диалог ➕")
                print("  /list - показать список диалогов 📝")
                print("  /select <id> - выбрать диалог по ID 🔍")
                print(
                    f"  Режим RAG: {'включен' if chat_bot.use_rag else 'выключен'} {'✅' if chat_bot.use_rag else '❌'}"
                )
                continue

            elif command == "rag":
                chat_bot.toggle_rag()
                print(
                    f"🔄 Режим RAG: {'включен' if chat_bot.use_rag else 'выключен'} {'✅' if chat_bot.use_rag else '❌'}"
                )
                continue

            elif command == "more_diverse":
                chat_bot.toggle_more_diverse()
                print("🎨 Режим разнообразных ответов включен, ответы могут быть менее точными")
                continue

            elif command == "more_accurate":
                chat_bot.toggle_more_accurate()
                print("🎯 Режим точных ответов включен, ответы могут быть менее разнообразными")
                continue

            elif command == "update_documents":
                chat_bot.update_documents()
                print("📚 Документы обновлены")
                continue

            elif command == "supported_extensions":
                print("📄 Поддерживаемые расширения:")
                print(chat_bot.get_supported_extensions())
                continue

            elif command == "new":
                conversation_id = chat_bot.create_new_conversation()
                print(f"✅ Новый диалог создан с ID: {conversation_id}")
                continue

            elif command == "list":
                conversations = chat_bot.get_conversations()
                if not conversations:
                    print("📝 Нет сохраненных диалогов")
                else:
                    print("\n📝 Список диалогов:")
                    for conversation in conversations:
                        print(f"  ID: {conversation['id']}, Создан: {conversation['created_at']}")
                continue

            elif command.startswith("select"):
                try:
                    select_id = command.split("select ")[1].strip()
                    if chat_bot.conversation_exists(select_id):
                        conversation_id = select_id
                        print(f"✅ Выбран диалог с ID: {conversation_id}")
                    else:
                        print(f"❌ Диалог с ID {select_id} не найден")
                except Exception:
                    print("⚠️ Неверный формат команды. Используйте: /select <id>")
                continue

            elif command.startswith("delete"):
                try:
                    delete_id = command.split("delete ")[1].strip()
                    success = chat_bot.delete_conversation(delete_id)
                    if success:
                        print(f"✅ Диалог с ID {delete_id} удален")
                        if delete_id == conversation_id:
                            conversation_id = chat_bot.create_new_conversation()
                            print(f"✅ Новый диалог создан с ID: {conversation_id}")
                    else:
                        print(f"❌ Диалог с ID {delete_id} не найден")
                except Exception as e:
                    print(f"⚠️ Ошибка при удалении диалога: {e}")
                continue

            else:
                print(f"❌ Неизвестная команда: {command}")
                continue

        response = chat_bot.chat(query, conversation_id)
        print(f"\n🤖 Бот: {response}")

    print("\n👋 До свидания!")


if __name__ == "__main__":
    main()
