"""
Request models for microservices (using existing response models from core)
"""
from pydantic import BaseModel, Field
from typing import List
from ..core.models import ExtractedKeywords, PriceAnalysisReport, NewsAnalysisReport

# Request Models Only
class SpeechInputRequest(BaseModel):
    duration: int = Field(7, description="Recording duration in seconds", ge=3, le=30)

class KeywordExtractionRequest(BaseModel):
    user_query: str = Field(..., description="User query to extract keywords from")

class PriceDataRequest(BaseModel):
    keywords: ExtractedKeywords = Field(..., description="Extracted keywords")

class NewsDataRequest(BaseModel):
    keywords: ExtractedKeywords = Field(..., description="Extracted keywords")

class RetrieveRequest(BaseModel):
    user_query: str = Field(..., description="User query for retrieval")
    keywords: ExtractedKeywords = Field(..., description="Extracted keywords")

class PriceAnalysisRequest(BaseModel):
    user_query: str = Field(..., description="User query")
    relevant_docs: List[dict] = Field(..., description="Relevant documents")

class NewsAnalysisRequest(BaseModel):
    user_query: str = Field(..., description="User query")
    relevant_docs: List[dict] = Field(..., description="Relevant documents")

class SynthesisRequest(BaseModel):
    user_query: str = Field(..., description="User query")
    keywords: ExtractedKeywords = Field(..., description="Extracted keywords")
    price_analysis: PriceAnalysisReport = Field(..., description="Price analysis report")
    news_analysis: NewsAnalysisReport = Field(..., description="News analysis report")

class SpeechInputResponse(BaseModel):
    user_query: str = Field(..., description="Transcribed user query")
    success: bool = Field(..., description="Success status")
    error: str = Field(None, description="Error message if any")

class DocumentResponse(BaseModel):
    documents: List[dict] = Field(..., description="List of documents")
    count: int = Field(..., description="Number of documents")

class RetrievalResponse(BaseModel):
    relevant_docs: List[dict] = Field(..., description="Retrieved relevant documents")
    count: int = Field(..., description="Number of retrieved documents")