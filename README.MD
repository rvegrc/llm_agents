## Описание проекта

**tg-bot** — это мультикомпонентный проект для работы с LLM-агентами, генерацией текста, эмбеддингами и интеграцией с Telegram-ботом. Проект включает backend-сервисы, frontend-бота, хранение данных  и запрос данных из RAG, а также jupiter ноутбуки для разработки и тестирования.

### Основные компоненты:

- **backend/** — серверная часть, реализующая API, обработку запросов, работу с LLM и памятью, хранение данных и векторных представлений.
- **frontend/bot/** — Telegram-бот, взаимодействующий с пользователем.
- **dev/** — Jupyter-ноутбуки и вспомогательные скрипты для тестирования и разработки.
- **docker-compose.yaml, .env** — файлы для запуска и конфигурации всех сервисов.

### Документация

[Использование переменных окружения](./doc/env.md)

[Использование API](./doc/api.md)

[Загрузка новых моделей](./doc/llm.md)

### Структура проекта

```text
tg-bot/
├── .env
├── docker-compose.yaml
├── README.MD
├── backend/
│   ├── api/
│   │   ├── .env
│   │   ├── api.py
│   │   ├── app.log
│   │   ├── Dockerfile
│   │   ├── langgraph_agent.py
│   │   ├── requirements.txt
│   │   └── tools/
│   │       ├── __init__.py
│   │       ├── llm.py
│   │       ├── memory.py
│   │       ├── rag.py
│   │       └── web_search.py
│   ├── data/
│   │   └── ollama_data/
│   │       ├── id_ed25519
│   │       └── id_ed25519.pub
│   └── vectorstore/
│       └── qdrant_data/
│           ├── raft_state.json
│           ├── aliases/
│           │   └── data.json
│           └── collections/
├── dev/
│   ├── api-test.ipynb
│   ├── bot-test.ipynb
│   ├── bot.py.bak
│   ├── langgraph_agent.ipynb
│   ├── llm.ipynb
│   ├── memory_tool.ipynb
│   ├── rag_tool.ipynb
│   └── search_tool.ipynb
└── frontend/
   └── bot/
       ├── .env
       ├── bot.py
       ├── Dockerfile
       └── requirements.txt
    
```