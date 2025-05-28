from langchain_core.messages import HumanMessage
from src.core.llm_config import get_llm
from src.core.models import ExtractedKeywords
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def extract_keywords_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts financial keywords from user query using LLM
    Args:
        state: Dictionary containing 'user_query' string
    Returns:
        Dictionary with 'extracted_keywords' (ExtractedKeywords)
    """
    try:
        if not state.get("user_query"):
            raise ValueError("No user query provided")
        
        llm = get_llm().with_structured_output(ExtractedKeywords)
        
        prompt = f"""
        Analyze this financial query and extract structured keywords:
        
        QUERY: {state["user_query"]}
        
        Extract:
        - Company names/tickers (e.g., AAPL, Microsoft)
        - Market sectors (e.g., tech, healthcare)
        - Geographic regions (e.g., US, Asia)
        - Financial terms (e.g., P/E ratio, volatility)
        - Time references (e.g., Q2 2023, last 6 months)
        
        Return ONLY the extracted keywords in structured format.
        """
        
        keywords = llm.invoke([HumanMessage(content=prompt)])
        logger.info(f"Extracted keywords: {keywords}")
        
        return {"extracted_keywords": keywords}
        
    except Exception as e:
        logger.error(f"Keyword extraction failed: {str(e)}")
        # Fallback with minimal keywords
        return {
            "extracted_keywords": ExtractedKeywords(
                companies=[], sectors=[], regions=[],
                financial_terms=[], time_references=[]
            )
        }