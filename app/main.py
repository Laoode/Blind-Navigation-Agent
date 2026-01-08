"""FastAPI application for blind navigation system.

Provides REST API endpoints for:
- Navigation guidance from camera input
- Health checks and system status
- Static file serving for frontend
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from agents.workflow import NavigationWorkflow
from app.config import Settings, get_settings
from app.exceptions import BlindNavException
from models.schemas import ProcessingRequest, ProcessingResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

workflow: NavigationWorkflow | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifecycle - initialize and cleanup resources."""
    global workflow
    
    try:
        settings = get_settings()
        workflow = NavigationWorkflow(settings)
        await workflow.initialize()
        logger.info("Application started successfully")
        yield
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise
    finally:
        if workflow:
            await workflow.cleanup()
        logger.info("Application shutdown complete")


app = FastAPI(
    title="Blind Navigation System",
    description="Multi-agent system for blind user navigation using FastVLM and YOLOv8",
    version="1.0.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


STATIC_DIR = Path(__file__).parent.parent / "static"
TEMPLATES_DIR = Path(__file__).parent.parent / "templates"


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main frontend page."""
    index_path = TEMPLATES_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(index_path)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "workflow_initialized": workflow is not None and workflow._initialized,
    }


@app.post("/api/navigate", response_model=ProcessingResponse)
async def navigate(request: ProcessingRequest):
    """Process navigation request with image and instruction.
    
    Args:
        request: Contains instruction text and optional base64 image.
        
    Returns:
        Navigation guidance with scene description and detected objects.
    """
    if workflow is None or not workflow._initialized:
        raise HTTPException(
            status_code=503,
            detail="Navigation system not initialized",
        )
    
    try:
        response = await workflow.process(request)
        return response
        
    except BlindNavException as e:
        logger.error(f"Navigation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    except Exception as e:
        logger.exception("Unexpected error during navigation")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-compatible chat completions endpoint.
    
    This endpoint provides compatibility with the frontend HTML reference,
    which expects an OpenAI-style API.
    """
    if workflow is None or not workflow._initialized:
        raise HTTPException(
            status_code=503,
            detail="Navigation system not initialized",
        )
    
    try:
        body = await request.json()
        messages = body.get("messages", [])
        
        instruction = "What do you see?"
        image_base64 = None
        
        for message in messages:
            content = message.get("content", [])
            if isinstance(content, str):
                instruction = content
            elif isinstance(content, list):
                for item in content:
                    if item.get("type") == "text":
                        instruction = item.get("text", instruction)
                    elif item.get("type") == "image_url":
                        image_url = item.get("image_url", {})
                        image_base64 = image_url.get("url", "")
        
        nav_request = ProcessingRequest(
            instruction=instruction,
            image_base64=image_base64,
        )
        
        response = await workflow.process(nav_request)
        
        return {
            "id": "nav-response",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response.guidance,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
        
    except Exception as e:
        logger.exception("Chat completions error")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=True,
    )