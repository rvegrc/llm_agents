# Переменные окружения (константы)

В проекте используются следующие переменные окружения, которые должны быть определены в файлах `.env` или в окружении системы:


- **API_TOKEN** — токен для авторизации запросов к API (используется в backend и frontend).
- **REQUIRE_API_TOKEN** — если установлено в `true`, API требует авторизации (по умолчанию `false`).
- **QDRANT_URL** — адрес сервера Qdrant (векторное хранилище).
- **QDRANT__SERVICE__API_KEY** — API-ключ для доступа к сервису Qdrant (если требуется авторизация на стороне Qdrant).
- **LLM_API_SERVER_URL** — адрес Ollama или другого LLM-сервера для генерации текста.
- **LLM_MODEL_NAME** — название используемой LLM-модели, поддерживающей tools (например, qwen3:4b, llama3.1 и др.).
- **BOT_TOKEN** — токен Telegram-бота (frontend/bot).
- **API_URL** — адрес backend API, к которому обращается бот.
- **LANGSMITH_PROJECT** — название проекта для интеграции с LangSmith (используется в некоторых инструментах памяти).
- **LANGSMITH_TRACING** — включает трассировку запросов в LangSmith ("true" для включения, по умолчанию выключено).
- **LANGSMITH_API_KEY** — API-ключ для доступа к сервису LangSmith (если используется интеграция).
- **TAVILY_API_KEY** — API-ключ для доступа к сервису Tavily (например, для внешнего поиска или дополнительных функций).


## Пример файла .env 
```env
API_TOKEN=your_api_token
REQUIRE_API_TOKEN=true
QDRANT_URL=http://qdrant:6333
LLM_API_SERVER_URL=http://ollama:11434
LLM_MODEL_NAME=qwen3:4b
BOT_TOKEN=your_telegram_token
API_URL=http://api:8000
LANGSMITH_PROJECT=tg-bot
LANGSMITH_TRACING="true"
LANGSMITH_API_KEY
TAVILY_API_KEY
```