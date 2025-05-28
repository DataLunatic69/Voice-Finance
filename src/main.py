from src.orchestration.workflow import create_workflow
from src.core.llm_config import get_llm
from src.services.speech_service import SpeechService
from src.core.models import AppState
import logging
import argparse

def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
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
    parser.add_argument(
        '--voice', 
        action='store_true',
        help='Process voice input from default microphone'
    )
    return parser.parse_args()

def main():
    configure_logging()
    logger = logging.getLogger(__name__)
    args = parse_args()
    
    # Initialize services
    services = {
        'workflow': create_workflow(),
        'speech_service': SpeechService(),
        'llm': get_llm()
    }
    
    # Process input
    user_query = args.query
    if args.voice and not user_query:
        logger.info("Starting voice recording...")
        audio_file = services['speech_service'].record_audio(duration=7)
        if audio_file:
            user_query = services['speech_service'].transcribe(audio_file)
    
    if not user_query:
        logger.error("No input provided")
        print("Please provide either --query or --voice argument")
        return
    
    logger.info(f"Processing query: {user_query}")
    
    # Execute workflow
    try:
        result = services['workflow'].invoke({
            "user_query": user_query,
            "price_documents": [],
            "news_documents": []
        })
        
        if result.get('final_report'):
            report = result['final_report']
            print("\n📊 FINAL ANALYSIS REPORT")
            print("=" * 40)
            print(f"\nExecutive Summary:\n{report.executive_summary}")
            print(f"\nRisk Assessment:\n{report.risk_exposure}")
            print("\nKey Highlights:")
            for highlight in report.key_highlights:
                print(f"- {highlight}")
            print("\nRecommendations:")
            for rec in report.recommendations:
                print(f"- {rec}")
        else:
            logger.error("Analysis failed to produce results")
            
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    main()