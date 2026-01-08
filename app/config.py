"""Application configuration using environment variables."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Settings:
    """Immutable application settings loaded from environment."""
    
    gemini_api_key: str
    gemini_model: str = "gemini-2.5-flash"
    
    fastvlm_model_path: str = "checkpoints/fastvlm_0.5b_stage3"
    yolo_model_path: str = "yolov8n-oiv7.pt"
    
    max_tokens: int = 256
    temperature: float = 0.3
    
    server_host: str = "0.0.0.0"
    server_port: int = 8000
    
    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment variables."""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        return cls(
            gemini_api_key=api_key,
            gemini_model=os.getenv("GEMINI_MODEL", cls.gemini_model),
            fastvlm_model_path=os.getenv("FASTVLM_MODEL_PATH", cls.fastvlm_model_path),
            yolo_model_path=os.getenv("YOLO_MODEL_PATH", cls.yolo_model_path),
            max_tokens=int(os.getenv("MAX_TOKENS", cls.max_tokens)),
            temperature=float(os.getenv("TEMPERATURE", cls.temperature)),
            server_host=os.getenv("SERVER_HOST", cls.server_host),
            server_port=int(os.getenv("SERVER_PORT", cls.server_port)),
        )


def get_settings() -> Settings:
    """Factory function for dependency injection."""
    return Settings.from_env()