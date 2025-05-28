from langchain_core.documents import Document
from src.core.vector_stores import price_vector_store
from src.core.models import AppState
from typing import List, Dict, Any

def price_retriever_agent(state: AppState) -> Dict[str, Any]:
    """Retrieve relevant price documents from vector store"""
    query = state["user_query"]
    keywords = state["extracted_keywords"]
    
    # Create enhanced query for price data
    enhanced_query = f"{query} price technical analysis {' '.join(keywords.companies)} {' '.join(keywords.financial_terms)}"
    
    # Retrieve documents
    price_retriever = price_vector_store.as_retriever(search_kwargs={"k": 10})
    relevant_docs = price_retriever.invoke(enhanced_query)
    
    return {"relevant_price_docs": relevant_docs}