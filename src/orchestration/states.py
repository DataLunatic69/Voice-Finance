from typing import TypedDict, List, Optional
from core.models import (
    ExtractedKeywords,
    PriceAnalysisReport,
    NewsAnalysisReport,
    FinalReport
)
from langchain_core.documents import Document

class AppState(TypedDict):
    """Main workflow state definition"""
    user_query: Optional[str]
    extracted_keywords: Optional[ExtractedKeywords]
    price_documents: List[Document]
    news_documents: List[Document]
    relevant_price_docs: List[Document]
    relevant_news_docs: List[Document]
    price_analysis_report: Optional[PriceAnalysisReport]
    news_analysis_report: Optional[NewsAnalysisReport]
    final_report: Optional[FinalReport]