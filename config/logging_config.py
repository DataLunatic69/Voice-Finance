import logging
from pathlib import Path
from config.settings import settings

def configure_logging():
    """Set up comprehensive logging configuration"""
    
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Base configuration
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "json": {
                "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                "fmt": "%(asctime)s %(name)s %(levelname)s %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "level": settings.LOG_LEVEL
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": logs_dir / "app.log",
                "maxBytes": 10 * 1024 * 1024,  # 10MB
                "backupCount": 5,
                "formatter": "standard",
                "level": settings.LOG_LEVEL
            },
            "json_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": logs_dir / "app.json.log",
                "maxBytes": 10 * 1024 * 1024,
                "backupCount": 5,
                "formatter": "json",
                "level": "DEBUG"
            }
        },
        "loggers": {
            "": {  # root logger
                "handlers": ["console", "file", "json_file"],
                "level": "DEBUG",
                "propagate": False
            },
            "uvicorn": {
                "handlers": ["console", "file"],
                "level": "INFO",
                "propagate": False
            },
            "httpx": {
                "handlers": ["file"],
                "level": "WARNING"
            }
        }
    }
    
    logging.config.dictConfig(logging_config)
    
    # Special library configurations
    logging.getLogger("openai").setLevel(logging.INFO)
    logging.getLogger("langchain").setLevel(logging.WARNING)