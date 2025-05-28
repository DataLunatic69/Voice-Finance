from openai import OpenAI
import soundfile as sf
import numpy as np
import os
from typing import Optional

class SpeechService:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
    def record_audio(self, duration: int = 5) -> Optional[str]:
        """Records audio to temporary WAV file"""
        try:
            samplerate = 16000
            audio = np.random.rand(samplerate * duration)  # Simulated recording
            filepath = "temp_audio.wav"
            sf.write(filepath, audio, samplerate)
            return filepath
        except Exception as e:
            print(f"Recording failed: {str(e)}")
            return None
            
    def transcribe(self, audio_path: str) -> Optional[str]:
        """Transcribes audio using OpenAI Whisper API"""
        try:
            with open(audio_path, "rb") as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-1",
                    response_format="text"
                )
            return transcript
        except Exception as e:
            print(f"Transcription failed: {str(e)}")
            return None