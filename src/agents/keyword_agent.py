from langchain_core.messages import HumanMessage
from core.llm_config import get_llm
from core.models import ExtractedKeywords
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
        user_query = state.get("user_query")
        logger.info(f"🔍 KEYWORD EXTRACTION STARTED")
        logger.debug(f"   Input query: '{user_query}'")
        
        if not user_query:
            raise ValueError("No user query provided")
        
        llm = get_llm().with_structured_output(ExtractedKeywords)
        logger.debug(f"   LLM initialized: {type(llm).__name__}")
        
        prompt = f"""
        Analyze this financial query and extract structured keywords:
        
        QUERY: {user_query}
        
        Extract:
        - Company names/tickers (e.g., AAPL, Microsoft)
        - Market sectors (e.g., tech, healthcare)
        - Geographic regions (e.g., US, Asia)
        - Financial terms (e.g., P/E ratio, volatility)
        - Time references (e.g., Q2 2023, last 6 months)
        
        Return ONLY the extracted keywords in structured format.
        """
        
        keywords = llm.invoke([HumanMessage(content=prompt)])
        logger.info(f"✅ KEYWORD EXTRACTION COMPLETED")
        logger.info(f"   Companies: {keywords.companies}")
        logger.info(f"   Sectors: {keywords.sectors}")
        logger.info(f"   Regions: {keywords.regions}")
        logger.info(f"   Financial Terms: {keywords.financial_terms}")
        logger.info(f"   Time References: {keywords.time_references}")
        
        return {"extracted_keywords": keywords}
        
    except Exception as e:
        logger.error(f"❌ Keyword extraction failed: {str(e)}")
        logger.warning(f"   Using fallback keywords for AAPL")
        # Fallback with default AAPL keywords
        fallback = ExtractedKeywords(
            companies=["AAPL"], 
            sectors=["technology"], 
            regions=["US"],
            financial_terms=["price", "analysis", "earnings"], 
            time_references=["today", "recent"]
        )
        return {
            "extracted_keywords": fallback
        }