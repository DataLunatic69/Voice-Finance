from langchain_core.documents import Document
from src.core.vector_stores import news_vector_store
from src.core.models import AppState
from typing import List, Dict, Any

def news_retriever_agent(state: AppState) -> Dict[str, Any]:
    """Retrieve relevant news documents from vector store"""
    query = state["user_query"]
    keywords = state["extracted_keywords"]
    
    # Create enhanced query for news data
    enhanced_query = f"{query} news earnings sentiment {' '.join(keywords.companies)} {' '.join(keywords.financial_terms)}"
    
    # Retrieve documents
    news_retriever = news_vector_store.as_retriever(search_kwargs={"k": 10})
    relevant_docs = news_retriever.invoke(enhanced_query)
    
    return {"relevant_news_docs": relevant_docs}