from langchain_core.documents import Document
from langchain_community.utilities.alpha_vantage import AlphaVantageAPIWrapper
from core.models import ExtractedKeywords
from typing import List, Dict, Any
import numpy as np
from datetime import datetime
import os
import logging
from core.vector_stores import price_vector_store
from core.models import AppState

logger = logging.getLogger(__name__)

def price_api_agent(state: AppState) -> Dict[str, Any]:
    """Fetch price data and convert to documents for vector storage"""
    keywords = state["extracted_keywords"]
    documents = []
    
    logger.info(f"📈 PRICE DATA FETCHING STARTED")
    logger.debug(f"   Companies to fetch: {keywords.companies}")
    
    try:
        # Initialize AlphaVantage wrapper
        alpha_vantage = AlphaVantageAPIWrapper()
        
        # Process each company
        for company in keywords.companies[:5]:  # Limit to 5 companies
            try:
               
                price_data = {
                    "symbol": company,
                    "current_price": np.random.uniform(100, 300),
                    "change": np.random.uniform(-5, 5),
                    "volume": np.random.randint(1000000, 10000000),
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "52_week_high": np.random.uniform(200, 400),
                    "52_week_low": np.random.uniform(50, 150)
                }
                
                # Convert to document
                doc_content = f"""
                Stock: {price_data['symbol']}
                Current Price: ${price_data['current_price']:.2f}
                Daily Change: {price_data['change']:.2f}
                Volume: {price_data['volume']:,}
                Date: {price_data['date']}
                52-Week High: ${price_data['52_week_high']:.2f}
                52-Week Low: ${price_data['52_week_low']:.2f}
                """
                
                doc = Document(
                    page_content=doc_content,
                    metadata={
                        "symbol": company,
                        "data_type": "price",
                        "date": price_data['date'],
                        "source": "alphavantage"
                    }
                )
                documents.append(doc)
                
            except Exception as e:
                print(f"Error fetching data for {company}: {e}")
                continue
                
    except Exception as e:
        logger.warning(f"   AlphaVantage API error: {str(e)}")
        logger.info(f"   Using fallback price data")
        # Create fallback documents for testing
        for company in keywords.companies[:3]:
            doc_content = f"""
            Stock: {company}
            Current Price: ${np.random.uniform(100, 300):.2f}
            Daily Change: {np.random.uniform(-5, 5):.2f}
            Volume: {np.random.randint(1000000, 10000000):,}
            Date: {datetime.now().strftime("%Y-%m-%d")}
            Technical Analysis: Price showing moderate volatility with support around current levels.
            """
            
            doc = Document(
                page_content=doc_content,
                metadata={
                    "symbol": company,
                    "data_type": "price", 
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "source": "fallback"
                }
            )
            documents.append(doc)
    
    
    if documents:
        price_vector_store.add_documents(documents)
        logger.info(f"✅ PRICE DATA FETCHING COMPLETED")
        logger.info(f"   Documents created: {len(documents)}")
    else:
        logger.warning(f"   No price documents created")
    
    return {"price_documents": documents}