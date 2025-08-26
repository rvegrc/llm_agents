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

from qdrant_client import QdrantClient, models

from datetime import datetime, timedelta, timezone

# from dotenv import load_dotenv
# load_dotenv()

from .rag import vectorstore_collection_init

# class agent with id  and thread do in api

import os
import requests


QDRANT_URL = os.getenv("QDRANT_URL")
LLM_API_SERVER_URL = os.getenv("LLM_API_SERVER_URL")

logging.info("Modules and Environment variables loaded.")
logging.info("Initializing Qdrant client.")

# Initialize Qdrant client
client_qd = QdrantClient(url=QDRANT_URL)

logging.info("Qdrant client initialized.")

# emb_model_name = '/models/multilingual-e5-large-instruct'
# embeddings = HuggingFaceEmbeddings(model_name=emb_model_name)

logging.info("Initializing embeddings.")

emb_model_name = 'nomic-embed-text'

embeddings = OllamaEmbeddings(
    base_url=LLM_API_SERVER_URL,
    model=emb_model_name
)

logging.info(f"Loaded embeddings model: {emb_model_name}")


recall_memories = vectorstore_collection_init(
    client_qd=client_qd,
    collection_name='recall_memories',
    embeddings=embeddings,
    distance="Cosine"
)

def get_user_thread_id(config: RunnableConfig) -> str:
    user_id = config["configurable"].get("user_id")
    thread_id = config["configurable"].get("thread_id")
    created_at = config["configurable"].get("created_at")
    if user_id is None:
        raise ValueError("User ID needs to be provided to save a memory.")
    if thread_id is None:
        raise ValueError("Thread ID needs to be provided to save a memory.")
    if created_at is None:
        raise ValueError("Message created at needs to be provided to save a memory.")

    return user_id, thread_id, created_at



def save_recall_memories(
        memory: str,
        config: RunnableConfig       
        ) -> str:
    """Save recall memory to vectorstore for later semantic retrieval."""
    user_id, thread_id, created_at = get_user_thread_id(config)
    # current timestamp
    document = Document(
        page_content=memory,
        metadata={
            "user_id": user_id,
            "thread_id": thread_id,
            "created_at": created_at
        }
    )
    recall_memories.add_documents([document])
    return memory


def search_recall_memories(
        query: str,
        config: RunnableConfig
    ) -> List[str]:
    """Search for relevant recall memories."""

    user_id, thread_id, created_at = get_user_thread_id(config)
    
    # Filter by user_id and thread_id
    qdrant_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.user_id",
                match=models.MatchValue(value=user_id),
            ),
            models.FieldCondition(
                key="metadata.thread_id",
                match=models.MatchValue(value=thread_id),
            ),
            # models.FieldCondition(
            #     key="metadata.created_at",
            #     range=models.Range(
            #         gte=(datetime.fromisoformat(created_at) - timedelta(days=30)).isoformat(),
            #         lte=datetime.fromisoformat(created_at).isoformat()
            #     )
            # )
        ]
    )

    documents = recall_memories.similarity_search(
        query,
        k=3,
        filter=qdrant_filter,  # structured filter required by QdrantVectorStore
    )
 

    # return [doc.page_content for doc in documents]
    return  [f'record: {doc.page_content}, this record created at: {doc.metadata.get("created_at")},'
              for doc in documents]

