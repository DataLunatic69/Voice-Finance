from langchain_core.documents import Document
from core.vector_stores import news_vector_store
from core.models import AppState
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

def news_retriever_agent(state: AppState) -> Dict[str, Any]:
    """Retrieve relevant news documents from vector store"""
    query = state["user_query"]
    keywords = state["extracted_keywords"]
    
    logger.info(f"🔎 NEWS DOCUMENT RETRIEVAL STARTED")
    logger.debug(f"   Query: '{query}'")
    logger.debug(f"   Available companies: {keywords.companies}")
    
    # Create enhanced query for news data
    enhanced_query = f"{query} news earnings sentiment {' '.join(keywords.companies)} {' '.join(keywords.financial_terms)}"
    logger.debug(f"   Enhanced query: '{enhanced_query}'")
    
    # Retrieve documents
    news_retriever = news_vector_store.as_retriever(search_kwargs={"k": 10})
    relevant_docs = news_retriever.invoke(enhanced_query)
    
    logger.info(f"✅ NEWS DOCUMENT RETRIEVAL COMPLETED")
    logger.info(f"   Documents retrieved: {len(relevant_docs)}")
    for i, doc in enumerate(relevant_docs[:3]):
        logger.debug(f"   Doc {i+1}: {doc.page_content[:100]}...")
    
    return {"relevant_news_docs": relevant_docs}