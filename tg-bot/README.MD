tg-bot/
 api/
    app.py
    Dockerfile
    requirements.txt
 bot/
    bot.py
    Dockerfile
    requirements.txt
 vector_store/
    qdrant_data/
 models/
    Qwen2.5-0.5B/
    Qwen2.5-7B-Instruct/
    multilingual-e5-large-instruct/
    Qwen3-Embedding-0.6B/

   
 docker-compose.yml
 .env

 text gen:
  - https://huggingface.co/unsloth/Qwen2.5-0.5B
  - https://huggingface.co/Qwen/Qwen3-0.6B
  - https://huggingface.co/unsloth/Qwen2.5-7B-Instruct
  - https://huggingface.co/openai/gpt-oss-20b
  - https://huggingface.co/Vikhrmodels/Vikhr-Qwen-2.5-0.5B-instruct-GGUF

 embeddings:
  - from https://huggingface.co/spaces/mteb/leaderboard
    - https://huggingface.co/intfloat/multilingual-e5-large-instruct
    - https://huggingface.co/Qwen/Qwen3-Embedding-0.6B


```scss
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
    ├── bot/
    │   ├── .env
    │   ├── bot.py
    │   ├── Dockerfile
    │   └── requirements.txt
    └── data/
        └── open-webui/
            ├── webui.db
            └── cache/
                ├── audio/
                │   └── speech/
                ├── embedding/
                │   └── models/
                │       ├── .locks/
                │       │   └── models--sentence-transformers--all-MiniLM-L6-v2/
                │       │       ├── ... (lock files)
                │       ├── models--sentence-transformers--all-MiniLM-L6-v2/
                │       │   ├── blobs/
                │       │   ├── refs/
                │       │   └── snapshots/
                │       └── xet/
                │           └── https___cas_serv-tGqkUaZf_CBPHQ6h/
                └── image/
                    └── generations/
            └── uploads/
```