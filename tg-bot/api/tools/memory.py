from typing import List
from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
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

# Initialize Qdrant client
client_qd = QdrantClient(url=QDRANT_URL)

# load from  backend
user_id = '123'
thread_id = '456'

emb_model_name = '../../../models/multilingual-e5-large-instruct'
embeddings = HuggingFaceEmbeddings(model_name=emb_model_name)



recall_memories_collection = vectorstore_collection_init(
    client_qd=client_qd,
    collection_name='recall_memories',
    embeddings=embeddings,
    distance="Cosine"
)



def save_recall_memories(
        memory: str       
        ) -> str:
    """Save recall memory to vectorstore for later semantic retrieval."""
    document = Document(
        page_content=memory, metadata={"user_id": user_id, 'thread_id': thread_id}
    )
    recall_memories_collection.add_documents([document])
    return memory


def search_recall_memories(
        query: str
) -> List[str]:
    """Search for relevant recall memories."""
    
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


