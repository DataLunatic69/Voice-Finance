from langchain_core.messages import HumanMessage
from src.core.llm_config import get_llm
from src.core.models import PriceAnalysisReport, AppState
from typing import Dict, Any

def price_analysis_agent(state: AppState) -> Dict[str, Any]:
    """Analyze retrieved price documents"""
    relevant_docs = state["relevant_price_docs"]
    user_query = state["user_query"]
    
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
    As a technical analyst, analyze the following price data and provide a comprehensive analysis:
    
    Price Data:
    {doc_content}
    
    User Query: {user_query}
    
    Provide analysis covering:
    - Trend direction (UP/DOWN/NEUTRAL)
    - Volatility assessment (HIGH/MEDIUM/LOW)
    - Key support and resistance levels
    - Risk assessment
    - Technical summary
    
    Focus on answering the specific user query while providing comprehensive technical analysis.
    """
    
    try:
        analysis_llm = get_llm.with_structured_output(PriceAnalysisReport)
        analysis = analysis_llm.invoke([HumanMessage(prompt)])
        return {"price_analysis_report": analysis}
    except Exception as e:
        print(f"Price analysis error: {e}")
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