from typing import Literal
from src.orchestration.states import AppState

def route_to_analysis(state: AppState) -> Literal["price", "news", "both"]:
    """Enhanced routing logic with sentiment analysis"""
    query = state["user_query"].lower()
    keywords = state["extracted_keywords"]
    
    has_price_terms = any(term in query for term in ["price", "technical", "chart"])
    has_news_terms = any(term in query for term in ["news", "sentiment", "earnings"])
    
    if has_price_terms and not has_news_terms:
        return "price"
    elif has_news_terms and not has_price_terms:
        return "news"
    return "both"