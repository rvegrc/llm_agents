from typing import List
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore

# class agent with id  and thread do in api

# load from  backend
user_id = '123'
thread_id = '456'

def save_memories(
        memory: str,
        vectorstore: QdrantVectorStore       
        ) -> str:
    """Save memory to vectorstore for later semantic retrieval."""
    document = Document(
        page_content=memory, metadata={"user_id": user_id, 'thread_id': thread_id}
    )
    vectorstore.add_documents([document])
    return memory


def search_memories(
        query: str, 
        vectorstore: QdrantVectorStore, 
                ) -> List[str]:
    """Search for relevant memories."""
    
    # Qdrant filter: match payload field "user_id" == user_id and "thread_id" == thread_id
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

    documents = vectorstore.similarity_search(
        query,
        k=3,
        filter=qdrant_filter,  # structured filter required by QdrantVectorStore
    )
    return [doc.page_content for doc in documents]


