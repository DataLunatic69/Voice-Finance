from langchain_core.messages import HumanMessage
from core.llm_config import get_llm
from core.models import PriceAnalysisReport, AppState
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def price_analysis_agent(state: AppState) -> Dict[str, Any]:
    """Analyze retrieved price documents"""
    relevant_docs = state["relevant_price_docs"]
    user_query = state["user_query"]
    
    logger.info(f"📊 PRICE ANALYSIS STARTED")
    logger.info(f"   Documents to analyze: {len(relevant_docs)}")
    
    if not get_llm:
        # Fallback for testing
        return {
            "price_analysis_report": PriceAnalysisReport(
                trend_direction="UP",
                volatility_level="MEDIUM",
                support_levels=[100.0, 95.0],
                resistance_levels=[110.0, 115.0],
                risk_assessment="Moderate risk with stable fundamentals",
                technical_summary="Price showing upward momentum with strong support levels"
            )
        }
    
    # Combine document content
    doc_content = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    prompt = f"""
    As a professional technical analyst, provide a DETAILED and COMPREHENSIVE price analysis:
    
    Price Data:
    {doc_content if doc_content.strip() else "No specific price data available - provide general technical analysis for the requested security"}
    
    User Query: {user_query}
    
    REQUIRED: Provide detailed analysis covering:
    - Trend direction (UP/DOWN/NEUTRAL) with explanation
    - Volatility assessment (HIGH/MEDIUM/LOW) and impact
    - Key support and resistance price levels with specific numbers
    - Comprehensive risk assessment and mitigation strategies
    - Detailed technical summary with patterns and signals
    
    Be specific, professional, and insightful. Provide actionable analysis."""
    
    try:
        analysis_llm = get_llm().with_structured_output(PriceAnalysisReport)
        logger.debug(f"   LLM initialized for analysis")
        analysis = analysis_llm.invoke([HumanMessage(content=prompt)])
        logger.info(f"✅ PRICE ANALYSIS COMPLETED")
        logger.info(f"   Trend: {analysis.trend_direction}")
        logger.info(f"   Volatility: {analysis.volatility_level}")
        logger.info(f"   Support Levels: {analysis.support_levels}")
        logger.info(f"   Resistance Levels: {analysis.resistance_levels}")
        return {"price_analysis_report": analysis}
    except Exception as e:
        logger.error(f"❌ Price analysis error: {str(e)}")
        # Return fallback analysis
        return {
            "price_analysis_report": PriceAnalysisReport(
                trend_direction="NEUTRAL",
                volatility_level="MEDIUM", 
                support_levels=[100.0],
                resistance_levels=[110.0],
                risk_assessment="Unable to complete full analysis",
                technical_summary="Limited technical analysis available"
            )
        }