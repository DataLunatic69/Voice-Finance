from langchain_core.messages import HumanMessage
from src.core.llm_config import get_llm
from src.core.models import NewsAnalysisReport, AppState
from typing import Dict, Any

def news_analysis_agent(state: AppState) -> Dict[str, Any]:
    """Analyze retrieved news documents"""
    relevant_docs = state["relevant_news_docs"]
    user_query = state["user_query"]
    
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
    As a news analyst, analyze the following financial news and provide comprehensive sentiment analysis:
    
    News Data:
    {doc_content}
    
    User Query: {user_query}
    
    Provide analysis covering:
    - Overall sentiment score (0-100, bearish to bullish)
    - Key market events identified
    - Any earnings surprises found
    - Overall market sentiment classification
    - Regional sentiment analysis if applicable
    
    Focus on answering the specific user query while providing comprehensive news analysis.
    """
    
    try:
        analysis_llm = get_llm.with_structured_output(NewsAnalysisReport)
        analysis = analysis_llm.invoke([HumanMessage(prompt)])
        return {"news_analysis_report": analysis}
    except Exception as e:
        print(f"News analysis error: {e}")
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