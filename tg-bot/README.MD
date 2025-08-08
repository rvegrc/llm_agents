
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


