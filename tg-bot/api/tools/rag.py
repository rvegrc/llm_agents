from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document


from typing import List


def add_qd_collection(
        documents: List[Document], 
        embeddings: HuggingFaceEmbeddings,
        client_qd: QdrantClient, 
        collection_name: str, 
        distance: str = "Cosine"
    ) -> None:
    
    ''' Add documents to a Qdrant collection. Creates the collection if it does not exist. Use HuggingFaceEmbeddings for embeddings.'''
        
    embedding_dim = SentenceTransformer(embeddings.model_name).get_sentence_embedding_dimension()

    # Create collection if it doesn't exist
    if not client_qd.collection_exists(collection_name):
        client_qd.create_collection(
            collection_name=collection_name,
            vectors_config={
                "vector": {
                    "size": embedding_dim,
                    "distance": distance,
                }
            },
        )


    # Initialize vector store
    qdrant = QdrantVectorStore(
        client=client_qd,
        collection_name=collection_name,
        embedding=embeddings,
        vector_name="vector"
    )

    # Add documents to the collection
    qdrant.add_documents(documents=documents)

def del_qd_collection(
        client_qd: QdrantClient, 
        collection_name: str
    ) -> None:
    
    ''' Drop a Qdrant collection if it exists. '''
    
    if client_qd.collection_exists(collection_name):
        client_qd.delete_collection(collection_name)
        print(f"Collection '{collection_name}' deleted.")
    else:
        print(f"Collection '{collection_name}' does not exist.")

# create a vector store for recall if it does not exist


# distance = "Cosine"  
# collection_name = 'recall_memories'

def rag_tool(
        client_qd: QdrantClient, 
        collection_name: str, 
        embeddings: HuggingFaceEmbeddings, 
        distance: str = "Cosine"
    ) -> QdrantVectorStore:
    
    ''' Initialize a vector store for recall memory. '''
    
    # Check if the collection exists
    if not client_qd.collection_exists(collection_name):
        add_qd_collection(
            documents=[],  # Start with an empty list, will add later
            embeddings=embeddings,
            client_qd=client_qd,
            collection_name=collection_name,
            distance=distance
        )

    return QdrantVectorStore(
        client=client_qd,
        collection_name=collection_name,
        embedding=embeddings,
        vector_name="vector"
    )
