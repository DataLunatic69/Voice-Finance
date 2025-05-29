"""
API endpoints for all agents
"""
from fastapi import APIRouter, HTTPException
from .request_model import (
    SpeechInputRequest, SpeechInputResponse,
    KeywordExtractionRequest, 
    PriceDataRequest, NewsDataRequest, DocumentResponse,
    RetrieveRequest, RetrievalResponse,
    PriceAnalysisRequest, NewsAnalysisRequest,
    SynthesisRequest
)
from ..core.models import ExtractedKeywords, PriceAnalysisReport, NewsAnalysisReport, FinalReport

# Import agent functions from correct locations
from src.agents.analysis_agents.news_analysis import news_analysis_agent
from src.agents.analysis_agents.price_analysis import price_analysis_agent
from src.agents.speech_agent import voice_agent
from src.agents.synthesis_agent import language_synthesis_agent
from src.agents.data_agents.news_api import news_api_agent
from src.agents.data_agents.price_api import price_api_agent
from src.agents.retriever_agents.news_retriever import news_retriever_agent
from src.agents.retriever_agents.price_retriever import price_retriever_agent
from src.agents.keyword_agent import extract_keywords_agent

router = APIRouter()

@router.post("/speech_input", response_model=SpeechInputResponse)
async def speech_input_endpoint(request: SpeechInputRequest):
    """Process speech input and return transcribed query"""
    try:
        state = {"user_query": ""}
        transcription = voice_agent(state)
        return SpeechInputResponse(
            user_query=transcription,
            success=True
        )
    except Exception as e:
        return SpeechInputResponse(
            user_query="",
            success=False,
            error=str(e)
        )

@router.post("/extract_keywords", response_model=ExtractedKeywords)
async def extract_keywords_endpoint(request: KeywordExtractionRequest):
    """Extract keywords from user query"""
    try:
        return extract_keywords_agent(request.user_query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/fetch_price_data", response_model=DocumentResponse)
async def fetch_price_data_endpoint(request: PriceDataRequest):
    """Fetch price data and return documents"""
    try:
        documents = price_api_agent(request.keywords)
        return DocumentResponse(
            documents=[{"page_content": doc.page_content, "metadata": doc.metadata} for doc in documents],
            count=len(documents)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/fetch_news_data", response_model=DocumentResponse)
async def fetch_news_data_endpoint(request: NewsDataRequest):
    """Fetch news data and return documents"""
    try:
        documents = news_api_agent(request.keywords)
        return DocumentResponse(
            documents=[{"page_content": doc.page_content, "metadata": doc.metadata} for doc in documents],
            count=len(documents)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/retrieve_price_docs", response_model=RetrievalResponse)
async def retrieve_price_docs_endpoint(request: RetrieveRequest):
    """Retrieve relevant price documents"""
    try:
        relevant_docs = price_retriever_agent(request.user_query, request.keywords)
        return RetrievalResponse(
            relevant_docs=[{"page_content": doc.page_content, "metadata": doc.metadata} for doc in relevant_docs],
            count=len(relevant_docs)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/retrieve_news_docs", response_model=RetrievalResponse)
async def retrieve_news_docs_endpoint(request: RetrieveRequest):
    """Retrieve relevant news documents"""
    try:
        relevant_docs = news_retriever_agent(request.user_query, request.keywords)
        return RetrievalResponse(
            relevant_docs=[{"page_content": doc.page_content, "metadata": doc.metadata} for doc in relevant_docs],
            count=len(relevant_docs)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze_price", response_model=PriceAnalysisReport)
async def analyze_price_endpoint(request: PriceAnalysisRequest):
    """Analyze price data and return analysis report"""
    try:
        from langchain_core.documents import Document
        docs = [Document(page_content=doc.get('page_content', ''), 
                        metadata=doc.get('metadata', {})) 
                for doc in request.relevant_docs]
        
        return price_analysis_agent(docs, request.user_query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze_news", response_model=NewsAnalysisReport)
async def analyze_news_endpoint(request: NewsAnalysisRequest):
    """Analyze news data and return analysis report"""
    try:
        from langchain_core.documents import Document
        docs = [Document(page_content=doc.get('page_content', ''), 
                        metadata=doc.get('metadata', {})) 
                for doc in request.relevant_docs]
        
        return news_analysis_agent(docs, request.user_query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/synthesize_report", response_model=FinalReport)
async def synthesize_report_endpoint(request: SynthesisRequest):
    """Synthesize final comprehensive report"""
    try:
        return language_synthesis_agent(
            request.price_analysis,
            request.news_analysis,
            request.user_query,
            request.keywords
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "agents_microservice"}