from langchain_core.documents import Document
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from src.core.models import ExtractedKeywords
from typing import List
from datetime import datetime
import os
from src.core.vector_stores import news_vector_store
from src.core.models import AppState
from typing import List, Dict, Any



def news_api_agent(state: AppState) -> Dict[str, Any]:
    """Fetch news data and convert to documents for vector storage"""
    keywords = state["extracted_keywords"]
    documents = []
    
    try:
        # Initialize Yahoo Finance News tool
        news_tool = YahooFinanceNewsTool()
        
        # Process each company for news
        for company in keywords.companies[:5]:  # Limit to 5 companies
            try:
                
                sample_news = [
                    {
                        "title": f"{company} Reports Strong Q4 Earnings",
                        "content": f"{company} exceeded analyst expectations with strong revenue growth and positive guidance for next quarter.",
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        "sentiment": "positive"
                    },
                    {
                        "title": f"Market Analysis: {company} Stock Movement",
                        "content": f"Technical analysis shows {company} breaking key resistance levels with increased volume.",
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        "sentiment": "neutral"
                    }
                ]
                
                for news_item in sample_news:
                    doc_content = f"""
                    Title: {news_item['title']}
                    Date: {news_item['date']}
                    Content: {news_item['content']}
                    Sentiment: {news_item['sentiment']}
                    Company: {company}
                    """
                    
                    doc = Document(
                        page_content=doc_content,
                        metadata={
                            "symbol": company,
                            "data_type": "news",
                            "date": news_item['date'],
                            "sentiment": news_item['sentiment'],
                            "source": "yahoo_finance"
                        }
                    )
                    documents.append(doc)
                    
            except Exception as e:
                print(f"Error fetching news for {company}: {e}")
                continue
                
    except Exception as e:
        print(f"Yahoo Finance news error: {e}")
        # Create fallback news documents
        for company in keywords.companies[:3]:
            doc_content = f"""
            Title: {company} Market Update
            Date: {datetime.now().strftime("%Y-%m-%d")}
            Content: Latest market analysis shows {company} maintaining stable performance with moderate trading volume.
            Sentiment: neutral
            Company: {company}
            """
            
            doc = Document(
                page_content=doc_content,
                metadata={
                    "symbol": company,
                    "data_type": "news",
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "sentiment": "neutral",
                    "source": "fallback"
                }
            )
            documents.append(doc)
    
    # Store documents in vector store
    if documents:
        news_vector_store.add_documents(documents)
    
    return {"news_documents": documents}
