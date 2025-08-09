from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document


from typing import List


def vectorstore_add_collection(
        embeddings: OllamaEmbeddings,
        client_qd: QdrantClient, 
        collection_name: str,
        distance: str = "Cosine"
    ) -> None:
    
    ''' Creates the collection if it does not exist. Use HuggingFaceEmbeddings for embeddings.''' 

    embedding_dim = len(embeddings.embed_query("test"))

    # for HuggingFaceEmbeddings
    # embedding_dim = SentenceTransformer(embeddings.model_name).get_sentence_embedding_dimension()

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

# distance = "Cosine"  
# collection_name = 'recall_memories'

def vectorstore_collection_init(
        client_qd: QdrantClient, 
        collection_name: str, 
        embeddings: OllamaEmbeddings,
        distance: str = "Cosine"
    ) -> QdrantVectorStore:
    
    ''' Initialize a vector store for memory. '''
    
    # Check if the collection exists
    if not client_qd.collection_exists(collection_name):
        vectorstore_add_collection(
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


def vectorstore_add_documents(
        client_qd: QdrantClient, 
        collection_name: str, 
        documents: List[Document], 
        embeddings: OllamaEmbeddings,        
    ) -> None:
    
    ''' Add documents to the Qdrant collection. '''
    
     # Initialize vectorstore collection
    vs_collection = vectorstore_collection_init(
        client_qd=client_qd,
        collection_name=collection_name,
        embeddings=embeddings,
        vector_name="vector",
    )


    # Add documents to the vectorstore collection
    vs_collection.add_documents(documents=documents)

def vectorstore_del_collection(
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



