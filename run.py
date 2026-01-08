"""Entry point for running the blind navigation server."""

import os
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn
from dotenv import load_dotenv

from app.config import get_settings


def main():
    """Run the FastAPI server."""
    load_dotenv()
    
    try:
        settings = get_settings()
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("\nPlease set required environment variables:")
        print("  export GEMINI_API_KEY='your-api-key'")
        sys.exit(1)
    
    print(f"Starting server on {settings.server_host}:{settings.server_port}")
    print(f"Using Gemini model: {settings.gemini_model}")
    print(f"FastVLM path: {settings.fastvlm_model_path}")
    print(f"YOLO path: {settings.yolo_model_path}")
    
    uvicorn.run(
        "app.main:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()