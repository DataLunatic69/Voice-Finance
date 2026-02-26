import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orchestration.workflow import create_workflow
from core.llm_config import get_llm
import logging
import argparse
from dotenv import load_dotenv

load_dotenv()

def configure_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('financial_analysis.log'),
            logging.StreamHandler()
        ]
    )

def parse_args():
    parser = argparse.ArgumentParser(description='Financial Analysis System')
    parser.add_argument(
        '--query', 
        type=str, 
        help='Financial query to analyze'
    )
    return parser.parse_args()

def main():
    configure_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info("🚀 FINANCIAL ANALYSIS SYSTEM STARTED")
    logger.info("="*60)
    
    # Check API keys
    logger.info("🔐 Checking API configuration...")
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if not groq_key:
        logger.error("❌ GROQ_API_KEY not configured in .env file")
        print("\n" + "="*60)
        print("❌ ERROR: GROQ_API_KEY is not configured!")
        print("="*60)
        print("\n📝 Setup Instructions:")
        print("  1. Get a FREE Groq API key from: https://console.groq.com/keys")
        print("  2. Copy your API key")
        print("  3. Create/edit .env file in project root with:")
        print("     GROQ_API_KEY=your_actual_key_here")
        print("  4. Run the app again")
        print("\n💡 Groq offers FREE API access with generous rate limits!")
        print("="*60 + "\n")
        return
    
    logger.info("✅ GROQ_API_KEY is configured")
    
    args = parse_args()
    
    # Initialize workflow
    logger.info("📋 Initializing workflow...")
    workflow = create_workflow()
    logger.info("✅ Workflow initialized successfully")
    
    # Process input
    user_query = args.query
    
    if not user_query:
        # Interactive text input if no --query arg
        print("\n" + "="*60)
        user_query = input("📝 Enter your financial query: ").strip()
        print("="*60 + "\n")
    
    if not user_query:
        logger.error("❌ No input provided")
        print("Please provide a query via --query argument or interactive input")
        return
    
    logger.info(f"📊 USER QUERY: '{user_query}'")
    logger.info("="*60)
    logger.info("🔄 STARTING WORKFLOW EXECUTION")
    logger.info("="*60)
    
    # Execute workflow
    try:
        result = workflow.invoke({
            "user_query": user_query,
            "price_documents": [],
            "news_documents": []
        })
        
        logger.info("="*60)
        logger.info("✅ WORKFLOW EXECUTION COMPLETED")
        logger.info("="*60)
        
        if result.get('final_report'):
            report = result['final_report']
            print("\n" + "="*60)
            print("📊 FINAL ANALYSIS REPORT")
            print("="*60)
            print(f"\n🎯 Executive Summary:\n{report.executive_summary}")
            print(f"\n⚠️  Risk Assessment:\n{report.risk_exposure}")
            print("\n📈 Key Highlights:")
            for i, highlight in enumerate(report.key_highlights, 1):
                print(f"  {i}. {highlight}")
            print("\n💡 Recommendations:")
            for i, rec in enumerate(report.recommendations, 1):
                print(f"  {i}. {rec}")
            print("\n🔮 Market Outlook:")
            print(f"  {report.market_outlook}")
            print("\n" + "="*60 + "\n")
            
            logger.info(f"📄 Final Report Generated Successfully")
        else:
            logger.error("❌ Analysis failed to produce results")
            print("\n❌ Analysis failed to produce results")
            
    except Exception as e:
        logger.error("="*60)
        logger.error(f"❌ WORKFLOW EXECUTION FAILED")
        logger.error(f"   Error: {str(e)}")
        logger.error("="*60)
        print(f"\n❌ Error during analysis: {str(e)}")
        print("   Check the log file for details: financial_analysis.log\n")

if __name__ == "__main__":
    main()