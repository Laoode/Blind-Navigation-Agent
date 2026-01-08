# Blind Navigation Multi-Agent System

Multi-agent system for blind user navigation using FastVLM-0.5B and YOLOv8n-OIV7, 
orchestrated by Gemini Flash 2.5.

## Architecture

```
                    +-------------------+
                    |    Frontend       |
                    |   (HTML/JS)       |
                    +--------+----------+
                             |
                             v
                    +--------+----------+
                    |    FastAPI        |
                    |    Server         |
                    +--------+----------+
                             |
                             v
              +-----------------------------+
              |     LangGraph Workflow      |
              +-----------------------------+
                    |              |
          +---------+              +---------+
          v                                  v
+------------------+              +------------------+
|  Scene Analyzer  |              | Object Detector  |
|   (FastVLM)      |              |   (YOLOv8)       |
+--------+---------+              +--------+---------+
          |                                |
          +----------------+---------------+
                           |
                           v
                 +------------------+
                 |   Orchestrator   |
                 |    (Gemini)      |
                 +------------------+
                           |
                           v
                 +------------------+
                 |   Navigation     |
                 |    Guidance      |
                 +------------------+
```

## Project Structure

```
blind_nav_system/
├── app/
│   ├── __init__.py
│   ├── config.py         # Configuration management
│   ├── exceptions.py     # Custom exceptions
│   └── main.py           # FastAPI application
├── agents/
│   ├── __init__.py
│   ├── base.py           # Base agent interface
│   ├── scene_analyzer.py # FastVLM integration
│   ├── object_detector.py # YOLOv8 integration
│   ├── orchestrator.py   # Gemini orchestration
│   └── workflow.py       # LangGraph workflow
├── models/
│   ├── __init__.py
│   └── schemas.py        # Pydantic models
├── templates/
│   └── index.html        # Frontend interface
├── static/               # Static assets
├── requirements.txt
├── run.py               # Entry point
├── .env.example
└── README.md
```

## Setup

### Prerequisites

- Python 3.10+
- MacBook with Apple Silicon (for FastVLM)
- Gemini API key

### Installation

```bash
# Clone and enter directory
cd blind_nav_system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download YOLOv8n-oiv7 model (auto-downloads on first run)
# Or manually: wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-oiv7.pt

# Set up environment
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### FastVLM Setup (Optional)

For actual FastVLM inference on Apple Silicon:

```bash
# Clone FastVLM repo
git clone https://github.com/apple/ml-fastvlm.git

# Download model
./ml-fastvlm/app/get_pretrained_mlx_model.sh --model 0.5b --dest checkpoints/

# Update .env
FASTVLM_MODEL_PATH=checkpoints/fastvlm_0.5b_stage3
```

## Running

```bash
# Set your API key
export GEMINI_API_KEY="your-api-key"

# Run the server
python run.py
```

Open http://localhost:8000 in your browser.

## API Endpoints

### POST /api/navigate
Primary navigation endpoint.

Request:
```json
{
  "instruction": "Describe what is in front of me",
  "image_base64": "data:image/jpeg;base64,..."
}
```

Response:
```json
{
  "guidance": "Move forward, clear path ahead",
  "scene_description": "Indoor corridor with fluorescent lighting",
  "objects_detected": 3,
  "processing_time_ms": 245.5
}
```

### POST /v1/chat/completions
OpenAI-compatible endpoint for frontend compatibility.

### GET /health
Health check endpoint.

## Development

### Phase 1 (Current - MVP)
- Text input/output via web interface
- Gemini orchestration working
- Mock scene analysis (FastVLM integration pending)
- YOLO object detection

### Phase 2 (Next)
- Speech-to-text input
- Text-to-speech output
- Full FastVLM integration

### Phase 3 (Future)
- Continuous video streaming
- Real-time hazard alerts
- Haptic feedback integration

## License

MIT