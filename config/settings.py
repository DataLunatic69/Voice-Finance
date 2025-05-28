import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Configuration
    OPENAI_API_KEY: str
    GROQ_API_KEY: str
    ALPHAVANTAGE_API_KEY: str
    
    # Application Settings
    LOG_LEVEL: str = "INFO"
    MAX_RETRIES: int = 3
    REQUEST_TIMEOUT: int = 30
    
    # Vector Store Configuration
    CHROMA_PERSIST_DIR: str = "data/chroma_stores"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    # Audio Settings
    AUDIO_SAMPLE_RATE: int = 16000
    MAX_AUDIO_DURATION: int = 15  # seconds
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"

def get_settings() -> Settings:
    """Initialize and return settings with environment validation"""
    return Settings()

# Singleton settings instance
settings = get_settings()