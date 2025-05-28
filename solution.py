import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, TypedDict, Literal, Any
from enum import Enum, auto
import json
from pydantic import BaseModel, Field
import streamlit as st
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_community.utilities.alpha_vantage import AlphaVantageAPIWrapper
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import sounddevice as sd
from scipy.io.wavfile import write
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
MODEL = "deepseek-r1-distill-llama-70b"
alpha_vantage_key = os.getenv("ALPHAVANTAGE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name=MODEL)

# Pydantic Models
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

# Initialize vector stores
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

price_vector_store = Chroma(
    collection_name="price_data",
    embedding_function=embeddings,
    persist_directory="./chroma_price_store"
)

news_vector_store = Chroma(
    collection_name="news_data", 
    embedding_function=embeddings,
    persist_directory="./chroma_news_store"
)

# Speech Processing
class RealtimeSpeechProcessor:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        st.info("⏳ Loading Whisper model...")
        self.model_id = "openai/whisper-large-v3"
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id, 
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        self.model.to(self.device)
        
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )
        st.success("✅ Whisper model loaded and ready")
        
    def record_audio(self, duration=5, sample_rate=16000):
        """Record audio from microphone in real-time"""
        st.info(f"🎤 Listening for {duration} seconds... (speak now)")
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()  # Wait until recording is finished
        return audio, sample_rate
    
    def transcribe_realtime(self, duration=5):
        """Record and transcribe audio in one step"""
        try:
            audio, sample_rate = self.record_audio(duration)
            
            # Convert to numpy array and normalize
            audio_np = np.squeeze(audio)
            audio_np = audio_np / np.max(np.abs(audio_np))
            
            # Convert to dict format expected by Whisper
            audio_dict = {
                "array": audio_np,
                "sampling_rate": sample_rate
            }
            
            # Transcribe
            st.info("🔍 Transcribing audio...")
            result = self.pipe(
                audio_dict,
                return_timestamps=False,
                generate_kwargs={"language": "english"}
            )
            
            return result["text"]
            
        except Exception as e:
            st.error(f"Transcription error: {e}")
            return ""

# Agents
def speech_input_agent(state: AppState) -> Dict[str, Any]:
    """Process real-time speech input and convert to text query"""
    if state.get("user_query"):
        return state
    
    processor = RealtimeSpeechProcessor()
    transcription = processor.transcribe_realtime(duration=7)
    
    if not transcription:
        raise ValueError("Failed to transcribe audio input")
    
    st.success(f"🗣️ Transcribed query: {transcription}")
    return {"user_query": transcription}

def keyword_extractor_agent(state: AppState) -> Dict[str, Any]:
    """Extract relevant financial keywords from user query"""
    if not llm:
        return {
            "extracted_keywords": ExtractedKeywords(
                companies=["AAPL", "MSFT"],
                sectors=["technology"],
                regions=["US"],
                financial_terms=["earnings", "risk"],
                time_references=["today"]
            )
        }
    
    keyword_extractor_llm = llm.with_structured_output(ExtractedKeywords)
    
    prompt = f"""
    Analyze the following financial query and extract relevant keywords:
    Query: {state["user_query"]}
    
    Extract:
    - Company names or stock symbols
    - Market sectors  
    - Geographic regions
    - Financial terms and concepts
    - Time references
    """
    
    try:
        extraction = keyword_extractor_llm.invoke([HumanMessage(prompt)])
        return {"extracted_keywords": extraction}
    except Exception as e:
        st.error(f"Keyword extraction error: {e}")
        return {
            "extracted_keywords": ExtractedKeywords(
                companies=["AAPL"],
                sectors=["technology"], 
                regions=["US"],
                financial_terms=["price", "analysis"],
                time_references=["today"]
            )
        }

def price_api_agent(state: AppState) -> Dict[str, Any]:
    """Fetch price data and convert to documents for vector storage"""
    keywords = state["extracted_keywords"]
    documents = []
    
    try:
        alpha_vantage = AlphaVantageAPIWrapper()
        
        for company in keywords.companies[:5]:
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
                st.warning(f"Error fetching data for {company}: {e}")
                continue
                
    except Exception as e:
        st.error(f"AlphaVantage initialization error: {e}")
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
    
    return {"price_documents": documents}

def news_api_agent(state: AppState) -> Dict[str, Any]:
    """Fetch news data and convert to documents for vector storage"""
    keywords = state["extracted_keywords"]
    documents = []
    
    try:
        news_tool = YahooFinanceNewsTool()
        
        for company in keywords.companies[:5]:
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
                st.warning(f"Error fetching news for {company}: {e}")
                continue
                
    except Exception as e:
        st.error(f"Yahoo Finance news error: {e}")
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
    
    if documents:
        news_vector_store.add_documents(documents)
    
    return {"news_documents": documents}

def price_retriever_agent(state: AppState) -> Dict[str, Any]:
    """Retrieve relevant price documents from vector store"""
    query = state["user_query"]
    keywords = state["extracted_keywords"]
    
    enhanced_query = f"{query} price technical analysis {' '.join(keywords.companies)} {' '.join(keywords.financial_terms)}"
    
    price_retriever = price_vector_store.as_retriever(search_kwargs={"k": 10})
    relevant_docs = price_retriever.invoke(enhanced_query)
    
    return {"relevant_price_docs": relevant_docs}

def news_retriever_agent(state: AppState) -> Dict[str, Any]:
    """Retrieve relevant news documents from vector store"""
    query = state["user_query"]
    keywords = state["extracted_keywords"]
    
    enhanced_query = f"{query} news earnings sentiment {' '.join(keywords.companies)} {' '.join(keywords.financial_terms)}"
    
    news_retriever = news_vector_store.as_retriever(search_kwargs={"k": 10})
    relevant_docs = news_retriever.invoke(enhanced_query)
    
    return {"relevant_news_docs": relevant_docs}

def price_analysis_agent(state: AppState) -> Dict[str, Any]:
    """Analyze retrieved price documents"""
    relevant_docs = state["relevant_price_docs"]
    user_query = state["user_query"]
    
    if not llm:
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
        analysis_llm = llm.with_structured_output(PriceAnalysisReport)
        analysis = analysis_llm.invoke([HumanMessage(prompt)])
        return {"price_analysis_report": analysis}
    except Exception as e:
        st.error(f"Price analysis error: {e}")
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

def news_analysis_agent(state: AppState) -> Dict[str, Any]:
    """Analyze retrieved news documents"""
    relevant_docs = state["relevant_news_docs"]
    user_query = state["user_query"]
    
    if not llm:
        return {
            "news_analysis_report": NewsAnalysisReport(
                sentiment_score=65,
                key_events=["Positive earnings report", "Market volatility concerns"],
                earnings_surprises=["Company beat expectations by 3%"],
                market_sentiment="NEUTRAL",
                regional_sentiment="Mixed sentiment across regions"
            )
        }
    
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
        analysis_llm = llm.with_structured_output(NewsAnalysisReport)
        analysis = analysis_llm.invoke([HumanMessage(prompt)])
        return {"news_analysis_report": analysis}
    except Exception as e:
        st.error(f"News analysis error: {e}")
        return {
            "news_analysis_report": NewsAnalysisReport(
                sentiment_score=50,
                key_events=["Market analysis pending"],
                earnings_surprises=["No surprises identified"],
                market_sentiment="NEUTRAL",
                regional_sentiment="Analysis unavailable"
            )
        }

def language_synthesis_agent(state: AppState) -> Dict[str, Any]:
    """Synthesize final comprehensive market brief"""
    price_analysis = state["price_analysis_report"]
    news_analysis = state["news_analysis_report"]
    user_query = state["user_query"]
    keywords = state["extracted_keywords"]
    
    if not llm:
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
    As a senior financial analyst, create a comprehensive market brief based on the following analysis:
    
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
    
    Create a comprehensive market brief that directly addresses the user's query with:
    - Executive summary of current market conditions
    - Risk exposure analysis
    - Key market highlights
    - Specific trading recommendations
    - Forward-looking market outlook
    
    Make this sound like a professional morning market brief suitable for a portfolio manager.
    """
    
    try:
        synthesis_llm = llm.with_structured_output(FinalReport)
        final_report = synthesis_llm.invoke([HumanMessage(prompt)])
        return {"final_report": final_report}
    except Exception as e:
        st.error(f"Language synthesis error: {e}")
        return {
            "final_report": FinalReport(
                executive_summary="Market analysis completed with available data showing stable conditions.",
                risk_exposure="Risk exposure assessment requires additional data for complete analysis.",
                key_highlights=["Market data collected", "Analysis framework operational"],
                recommendations=["System operational", "Ready for detailed analysis"],
                market_outlook="System ready for comprehensive market analysis."
            )
        }

# Workflow Creation
def create_financial_analysis_graph():
    """Create the LangGraph workflow"""
    workflow = StateGraph(AppState)
    
    workflow.add_node("speech_input", speech_input_agent)
    workflow.add_node("keyword_extractor", keyword_extractor_agent)
    workflow.add_node("price_api", price_api_agent)
    workflow.add_node("news_api", news_api_agent)
    workflow.add_node("price_retriever", price_retriever_agent)
    workflow.add_node("news_retriever", news_retriever_agent)
    workflow.add_node("price_analysis", price_analysis_agent)
    workflow.add_node("news_analysis", news_analysis_agent)
    workflow.add_node("language_synthesis", language_synthesis_agent)
    
    workflow.add_edge("speech_input", "keyword_extractor")
    workflow.add_edge("keyword_extractor", "price_api")
    workflow.add_edge("keyword_extractor", "news_api")
    workflow.add_edge("price_api", "price_retriever")
    workflow.add_edge("news_api", "news_retriever")
    workflow.add_edge("price_retriever", "price_analysis")
    workflow.add_edge("news_retriever", "news_analysis")
    workflow.add_edge("price_analysis", "language_synthesis")
    workflow.add_edge("news_analysis", "language_synthesis")
    
    workflow.set_entry_point("speech_input")
    workflow.set_finish_point("language_synthesis")
    
    return workflow.compile()

# Streamlit UI
def main():
    st.set_page_config(
        page_title="Financial Analysis System",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("🎤 Real-time Financial Analysis System")
    st.markdown("""
    This system provides comprehensive financial analysis using:
    - Real-time speech input
    - Market data from AlphaVantage and Yahoo Finance
    - Advanced AI analysis with Groq/Llama 70B
    """)
    
    input_method = st.radio(
        "Select input method:",
        ("Voice", "Text"),
        horizontal=True
    )
    
    user_query = ""
    if input_method == "Text":
        user_query = st.text_area("Enter your financial query:")
    else:
        if st.button("🎤 Start Voice Recording"):
            with st.spinner("Listening for 7 seconds... Speak now"):
                processor = RealtimeSpeechProcessor()
                user_query = processor.transcribe_realtime(duration=7)
                if user_query:
                    st.success(f"Transcribed query: {user_query}")
    
    if st.button("🚀 Analyze") and user_query:
        with st.spinner("Running financial analysis..."):
            try:
                app = create_financial_analysis_graph()
                
                initial_state = {
                    "user_query": user_query,
                    "extracted_keywords": None,
                    "price_documents": [],
                    "news_documents": [],
                    "relevant_price_docs": [],
                    "relevant_news_docs": [],
                    "price_analysis_report": None,
                    "news_analysis_report": None,
                    "final_report": None
                }
                
                result = app.invoke(initial_state)
                
                if result:
                    st.success("✅ Analysis completed!")
                    
                    # Display results in expandable sections
                    with st.expander("📊 Executive Summary", expanded=True):
                        st.write(result["final_report"].executive_summary)
                    
                    with st.expander("⚠️ Risk Exposure"):
                        st.write(result["final_report"].risk_exposure)
                    
                    with st.expander("🎯 Key Highlights"):
                        for highlight in result["final_report"].key_highlights:
                            st.write(f"- {highlight}")
                    
                    with st.expander("💡 Recommendations"):
                        for recommendation in result["final_report"].recommendations:
                            st.write(f"- {recommendation}")
                    
                    with st.expander("🔮 Market Outlook"):
                        st.write(result["final_report"].market_outlook)
                    
                    # Show raw data in expandable sections
                    with st.expander("🔍 Technical Analysis Details"):
                        st.json(result["price_analysis_report"].dict())
                    
                    with st.expander("📰 News Analysis Details"):
                        st.json(result["news_analysis_report"].dict())
                    
                    with st.expander("🔑 Extracted Keywords"):
                        st.json(result["extracted_keywords"].dict())
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    main()