from langchain_core.messages import HumanMessage
from core.llm_config import get_llm
from core.models import NewsAnalysisReport, AppState
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def news_analysis_agent(state: AppState) -> Dict[str, Any]:
    """Analyze retrieved news documents"""
    relevant_docs = state["relevant_news_docs"]
    user_query = state["user_query"]
    
    logger.info(f"📰 NEWS ANALYSIS STARTED")
    logger.info(f"   Documents to analyze: {len(relevant_docs)}")
    
    if not get_llm:
        # Fallback for testing
        return {
            "news_analysis_report": NewsAnalysisReport(
                sentiment_score=65,
                key_events=["Positive earnings report", "Market volatility concerns"],
                earnings_surprises=["Company beat expectations by 3%"],
                market_sentiment="NEUTRAL",
                regional_sentiment="Mixed sentiment across regions"
            )
        }
    
    # Combine document content
    doc_content = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    prompt = f"""
    As a senior financial news analyst, provide DETAILED and COMPREHENSIVE sentiment analysis:
    
    News Data:
    {doc_content if doc_content.strip() else "No specific news data available - provide general sentiment analysis based on the query"}
    
    User Query: {user_query}
    
    REQUIRED: Provide detailed analysis covering:
    - Overall sentiment score (0-100, 0=extremely bearish, 100=extremely bullish) with justification
    - Key market events identified and their impact
    - Any earnings surprises, guidance changes, or major announcements
    - Overall market sentiment classification with reasoning
    - Regional sentiment analysis if applicable
    
    Be specific, detailed, and data-driven. Explain your sentiment reasoning clearly."""
    
    try:
        analysis_llm = get_llm().with_structured_output(NewsAnalysisReport)
        logger.debug(f"   LLM initialized for analysis")
        analysis = analysis_llm.invoke([HumanMessage(content=prompt)])
        logger.info(f"✅ NEWS ANALYSIS COMPLETED")
        logger.info(f"   Sentiment Score: {analysis.sentiment_score}/100")
        logger.info(f"   Market Sentiment: {analysis.market_sentiment}")
        logger.info(f"   Key Events: {analysis.key_events}")
        logger.info(f"   Earnings Surprises: {analysis.earnings_surprises}")
        return {"news_analysis_report": analysis}
    except Exception as e:
        logger.error(f"❌ News analysis error: {str(e)}")
        # Return fallback analysis
        return {
            "news_analysis_report": NewsAnalysisReport(
                sentiment_score=50,
                key_events=["Market analysis pending"],
                earnings_surprises=["No surprises identified"],
                market_sentiment="NEUTRAL",
                regional_sentiment="Analysis unavailable"
            )
        }