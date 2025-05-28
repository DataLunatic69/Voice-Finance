from langgraph.graph import StateGraph
from src.orchestration.states import AppState
from src.agents.analysis_agents.news_analysis import news_analysis_agent
from src.agents.analysis_agents.price_analysis import price_analysis_agent
from src.agents.speech_agent import voice_agent
from src.agents.synthesis_agent import language_synthesis_agent
from src.agents.data_agents.news_api import news_api_agent
from src.agents.data_agents.price_api import price_api_agent
from src.agents.retriever_agents.news_retriever import news_retriever_agent
from src.agents.retriever_agents.price_retriever import price_retriever_agent

def create_workflow() -> StateGraph:
    """Builds the complete LangGraph workflow"""
    workflow = StateGraph(AppState)
    
    # Add all nodes
    workflow.add_node("speech_input", voice_agent)
    workflow.add_node("keyword_extraction", keyword_agent.process)
    workflow.add_node("price_data", price_api_agent)
    workflow.add_node("news_data", news_api_agent)
    workflow.add_node("price_retrieval", price_retriever_agent)
    workflow.add_node("news_retrieval", news_retriever_agent)
    workflow.add_node("price_analysis", price_analysis_agent)
    workflow.add_node("news_analysis", news_analysis_agent)
    workflow.add_node("report_synthesis", language_synthesis_agent)
    
    # Define workflow edges
    workflow.set_entry_point("speech_input")
    workflow.add_edge("speech_input", "keyword_extraction")
    workflow.add_edge("keyword_extraction", "price_data")
    workflow.add_edge("keyword_extraction", "news_data")
    workflow.add_edge("price_data", "price_retrieval")
    workflow.add_edge("news_data", "news_retrieval")
    workflow.add_edge("price_retrieval", "price_analysis")
    workflow.add_edge("news_retrieval", "news_analysis")
    workflow.add_edge("price_analysis", "report_synthesis")
    workflow.add_edge("news_analysis", "report_synthesis")
    workflow.set_finish_point("report_synthesis")
    
    return workflow.compile()