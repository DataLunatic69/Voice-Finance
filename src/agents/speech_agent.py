from src.services.speech_service import SpeechService
from src.core.models import AppState
from typing import Dict, Any

def voice_agent(state: AppState) -> Dict[str, Any]:
    """Agent interface for voice processing"""
    if state.get("user_query"):  # Skip if text input already exists
        return state
        
    service = SpeechService()
    audio_file = service.record_audio(duration=5)
    
    if not audio_file:
        raise ValueError("Audio recording failed")
    
    transcription = service.transcribe(audio_file)
    
    if not transcription:
        raise ValueError("Transcription failed")
    
    return {"user_query": transcription}