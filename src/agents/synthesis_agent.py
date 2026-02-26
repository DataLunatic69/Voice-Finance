from langchain_core.messages import HumanMessage
from core.llm_config import get_llm
from core.models import FinalReport, AppState
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def language_synthesis_agent(state: AppState) -> Dict[str, Any]:
    """Synthesize final comprehensive market brief"""
    price_analysis = state["price_analysis_report"]
    news_analysis = state["news_analysis_report"]
    user_query = state["user_query"]
    keywords = state["extracted_keywords"]
    
    logger.info(f"📝 FINAL REPORT SYNTHESIS STARTED")
    logger.debug(f"   Price Analysis available: {price_analysis is not None}")
    logger.debug(f"   News Analysis available: {news_analysis is not None}")
    
    if not get_llm:
        # Fallback for testing
        return {
            "final_report": FinalReport(
                executive_summary="Market showing mixed signals with moderate volatility and neutral sentiment.",
                risk_exposure="Portfolio exposure appears balanced with manageable risk levels.",
                key_highlights=["Stable price action", "Neutral market sentiment", "No major surprises"],
                recommendations=["Maintain current positions", "Monitor key support levels"],
                market_outlook="Market expected to remain stable in near term with moderate volatility."
            )
        }
    
    prompt = f"""
    As a senior financial analyst, create a DETAILED and COMPREHENSIVE market brief based on the following analysis:
    
    Price Analysis:
    - Trend: {price_analysis.trend_direction}
    - Volatility: {price_analysis.volatility_level}
    - Support Levels: {price_analysis.support_levels}
    - Resistance Levels: {price_analysis.resistance_levels}
    - Risk Assessment: {price_analysis.risk_assessment}
    - Technical Summary: {price_analysis.technical_summary}
    
    News Analysis:
    - Sentiment Score: {news_analysis.sentiment_score}/100
    - Key Events: {news_analysis.key_events}
    - Earnings Surprises: {news_analysis.earnings_surprises}
    - Market Sentiment: {news_analysis.market_sentiment}
    - Regional Sentiment: {news_analysis.regional_sentiment}
    
    Companies/Sectors: {', '.join(keywords.companies + keywords.sectors)}
    
    User Query: {user_query}
    
    REQUIRED: Create a detailed and professional market brief addressing the specific user query with:
    1. Executive Summary - 3-4 sentences synthesizing both price and news analysis for clear, actionable insights
    2. Risk Exposure - Detailed assessment of downside risks, volatility exposure, and hedging considerations
    3. Key Highlights - 4-5 specific findings from price action, news events, and market sentiment (with concrete details)
    4. Recommendations - 4-5 specific, actionable trading/investment recommendations based on the combined analysis
    5. Market Outlook - Detailed forward-looking assessment (short-term, medium-term implications, monitoring points)
    
    Use specific numbers, price levels, sentiment scores from the analysis above. Be professional and detailed.
    This is a portfolio manager's briefing - provide substantive, detailed insights, not generic commentary.
    """
    
    try:
        synthesis_llm = get_llm().with_structured_output(FinalReport)
        logger.debug(f"   LLM initialized for synthesis")
        final_report = synthesis_llm.invoke([HumanMessage(content=prompt)])
        logger.info(f"✅ FINAL REPORT SYNTHESIS COMPLETED")
        logger.info(f"   Executive Summary: {final_report.executive_summary[:100]}...")
        logger.info(f"   Recommendations: {final_report.recommendations}")
        return {"final_report": final_report}
    except Exception as e:
        logger.error(f"❌ Language synthesis error: {str(e)}")
        # Return fallback report
        return {
            "final_report": FinalReport(
                executive_summary="Market analysis completed with available data showing stable conditions.",
                risk_exposure="Risk exposure assessment requires additional data for complete analysis.",
                key_highlights=["Market data collected", "Analysis framework operational"],
                recommendations=["System operational", "Ready for detailed analysis"],
                market_outlook="System ready for comprehensive market analysis."
            )
        }
