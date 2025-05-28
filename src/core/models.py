from pydantic import BaseModel, Field
from typing import List, Literal, Dict, Any, Optional, TypedDict
from datetime import datetime
from langchain_core.documents import Document

class ExtractedKeywords(BaseModel):
    """Keywords extracted from user query"""
    companies: List[str] = Field(description="Company names or stock symbols mentioned")
    sectors: List[str] = Field(description="Market sectors mentioned (e.g., tech, healthcare)")
    regions: List[str] = Field(description="Geographic regions mentioned (e.g., Asia, US)")
    financial_terms: List[str] = Field(description="Financial terms mentioned (e.g., earnings, volatility)")
    time_references: List[str] = Field(description="Time periods mentioned (e.g., today, this week)")

class PriceAnalysisReport(BaseModel):
    """Price analysis report structure"""
    trend_direction: Literal["UP", "DOWN", "NEUTRAL"] = Field(description="Overall price trend")
    volatility_level: Literal["HIGH", "MEDIUM", "LOW"] = Field(description="Current volatility assessment")
    support_levels: List[float] = Field(description="Key support price levels")
    resistance_levels: List[float] = Field(description="Key resistance price levels")
    risk_assessment: str = Field(description="Risk assessment summary")
    technical_summary: str = Field(description="Technical analysis summary")

class NewsAnalysisReport(BaseModel):
    """News analysis report structure"""
    sentiment_score: int = Field(description="Sentiment score 0-100 (bearish to bullish)")
    key_events: List[str] = Field(description="Important market events identified")
    earnings_surprises: List[str] = Field(description="Earnings surprises found")
    market_sentiment: Literal["BULLISH", "BEARISH", "NEUTRAL"] = Field(description="Overall market sentiment")
    regional_sentiment: str = Field(description="Regional sentiment analysis")

class FinalReport(BaseModel):
    """Final comprehensive market brief"""
    executive_summary: str = Field(description="Executive summary of market conditions")
    risk_exposure: str = Field(description="Portfolio risk exposure analysis")
    key_highlights: List[str] = Field(description="Key market highlights")
    recommendations: List[str] = Field(description="Trading recommendations")
    market_outlook: str = Field(description="Forward-looking market outlook")

class AppState(TypedDict):
    # Input processing
    user_query: str
    extracted_keywords: ExtractedKeywords
    
    # API agent outputs (documents for vector store)
    price_documents: List[Document]
    news_documents: List[Document]
    
    # Retriever outputs  
    relevant_price_docs: List[Document]
    relevant_news_docs: List[Document]
    
    # Analysis outputs
    price_analysis_report: PriceAnalysisReport
    news_analysis_report: NewsAnalysisReport
    
    # Final output
    final_report: FinalReport
