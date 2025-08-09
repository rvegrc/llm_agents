import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log"),          # File output
        logging.StreamHandler()                  # Console output
    ],
    force=True  # This overrides any prior logging config
)

logging.getLogger().info("Logging is set up.")

logging.info("Importing necessary modules for the application and load environment variables.")


from typing import List
from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_core.runnables import RunnableConfig

from qdrant_client import QdrantClient

from tools.rag import vectorstore_collection_init

# class agent with id  and thread do in api


import os
import requests
from dotenv import load_dotenv
load_dotenv()

# Initialize LangSmith project
os.environ["LANGSMITH_PROJECT"] = 'tg-bot'

QDRANT_URL = os.getenv("QDRANT_URL")
LLM_API_SERVER_URL = os.getenv("LLM_API_SERVER_URL")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")

logging.info("Modules and Environment variables loaded.")
logging.info("Initializing Qdrant client.")

# Initialize Qdrant client
client_qd = QdrantClient(url=QDRANT_URL)

logging.info("Qdrant client initialized.")

# emb_model_name = '/models/multilingual-e5-large-instruct'
# embeddings = HuggingFaceEmbeddings(model_name=emb_model_name)

logging.info("Using Ollama embeddings.")

emb_model_name = 'nomic-embed-text'

embeddings = OllamaEmbeddings(
    base_url=f'http://ollama:11434',
    model=emb_model_name
)

logging.info(f"Using embeddings model: {emb_model_name}")


recall_memories_collection = vectorstore_collection_init(
    client_qd=client_qd,
    collection_name='recall_memories',
    embeddings=embeddings,
    distance="Cosine"
)

def get_user_thread_id(config: RunnableConfig) -> str:
    user_id = config["configurable"].get("user_id")
    thread_id = config["configurable"].get("thread_id")
    if user_id is None:
        raise ValueError("User ID needs to be provided to save a memory.")
    if thread_id is None:
        raise ValueError("Thread ID needs to be provided to save a memory.")

    return user_id, thread_id



def save_recall_memories(
        memory: str,
        config: RunnableConfig       
        ) -> str:
    """Save recall memory to vectorstore for later semantic retrieval."""
    user_id, thread_id = get_user_thread_id(config)
    document = Document(
        page_content=memory, metadata={"user_id": user_id, 'thread_id': thread_id}
    )
    recall_memories_collection.add_documents([document])
    return memory


def search_recall_memories(
        query: str,
        config: RunnableConfig
) -> List[str]:
    """Search for relevant recall memories."""

    user_id, thread_id = get_user_thread_id(config)
    
    # Filter by user_id and thread_id
    qdrant_filter = {
        "must": [
            {
                "key": "user_id",
                "match": {"value": user_id},
            },
            {
                "key": "thread_id",
                "match": {"value": thread_id},
            },
        ]
    }

    documents = recall_memories_collection.similarity_search(
        query,
        k=3,
        filter=qdrant_filter,  # structured filter required by QdrantVectorStore
    )
    return [doc.page_content for doc in documents]


