import streamlit as st
from orchestration.workflow import create_workflow
from core.models import AppState
from core.llm_config import get_llm
from dotenv import load_dotenv
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_services():
    """Initialize all required services"""
    return {
        'workflow': create_workflow(),
        'llm': get_llm()
    }

def show_analysis_result(result: AppState):
    """Display analysis results in Streamlit"""
    if not result.get('final_report'):
        st.error("Analysis failed to produce results")
        return

    report = result['final_report']
    
    with st.expander("📊 Executive Summary", expanded=True):
        st.write(report.executive_summary)
    
    with st.expander("⚠️ Risk Assessment"):
        st.write(report.risk_exposure)
    
    with st.expander("🔍 Key Highlights"):
        for highlight in report.key_highlights:
            st.markdown(f"- {highlight}")
    
    with st.expander("💡 Recommendations"):
        for recommendation in report.recommendations:
            st.markdown(f"- {recommendation}")

def main():
    st.set_page_config(
        page_title="Financial Analysis System",
        page_icon="📈",
        layout="wide"
    )
    
    services = initialize_services()
    
    st.title("AI Financial Analyst")
    st.markdown("""
    Get real-time market analysis through text input.
    """)
    
    # Text input only
    user_query = st.text_area(
        "Enter your financial query:",
        placeholder="e.g. Analyze AAPL stock price and recent news"
    )
    
    if user_query:
        st.info(f"Processing query: {user_query}")
        
        if st.button("Analyze", type="primary"):
            with st.spinner("Running analysis..."):
                try:
                    result = services['workflow'].invoke({
                        "user_query": user_query,
                        "price_documents": [],
                        "news_documents": []
                    })
                    show_analysis_result(result)
                except Exception as e:
                    logger.error(f"Analysis failed: {str(e)}")
                    st.error("Analysis failed. Please try again.")

if __name__ == "__main__":
    main()