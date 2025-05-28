from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os
from typing import Tuple

def initialize_vector_stores() -> Tuple[Chroma, Chroma]:
    """Initialize and return both vector stores with persistent storage"""
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    price_store = Chroma(
        collection_name="price_data",
        embedding_function=embeddings,
        persist_directory=os.path.join("data", "chroma_price_store")
    )
    
    news_store = Chroma(
        collection_name="news_data",
        embedding_function=embeddings, 
        persist_directory=os.path.join("data", "chroma_news_store")
    )
    
    return price_store, news_store

# Initialize stores on import
price_vector_store, news_vector_store = initialize_vector_stores()