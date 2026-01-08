"""Pydantic models for request/response validation and internal data structures."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class AgentType(str, Enum):
    """Types of specialized agents in the system."""
    SCENE_ANALYZER = "scene_analyzer"
    OBJECT_DETECTOR = "object_detector"
    ORCHESTRATOR = "orchestrator"


class DetectedObject(BaseModel):
    """Single detected object from YOLO."""
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: tuple[float, float, float, float]
    position_description: str = ""


class SceneAnalysis(BaseModel):
    """Output from FastVLM scene analysis."""
    description: str
    spatial_context: str = ""
    potential_hazards: list[str] = Field(default_factory=list)


class NavigationContext(BaseModel):
    """Combined context from all agents for navigation guidance."""
    scene_analysis: Optional[SceneAnalysis] = None
    detected_objects: list[DetectedObject] = Field(default_factory=list)
    raw_detections_count: int = 0


class NavigationGuidance(BaseModel):
    """Final navigation guidance for the user."""
    primary_instruction: str
    spatial_awareness: str = ""
    hazard_warnings: list[str] = Field(default_factory=list)
    confidence_level: str = "medium"


class ProcessingRequest(BaseModel):
    """Input request from the frontend."""
    instruction: str = Field(default="What do you see?")
    image_base64: Optional[str] = None


class ProcessingResponse(BaseModel):
    """Output response to the frontend."""
    guidance: str
    scene_description: str = ""
    objects_detected: int = 0
    processing_time_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class AgentState:
    """Mutable state passed through the LangGraph workflow."""
    
    instruction: str = ""
    image_base64: Optional[str] = None
    
    scene_analysis: Optional[SceneAnalysis] = None
    detected_objects: list[DetectedObject] = field(default_factory=list)
    
    navigation_guidance: Optional[NavigationGuidance] = None
    
    errors: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for LangGraph compatibility."""
        return {
            "instruction": self.instruction,
            "image_base64": self.image_base64,
            "scene_analysis": self.scene_analysis.model_dump() if self.scene_analysis else None,
            "detected_objects": [obj.model_dump() for obj in self.detected_objects],
            "navigation_guidance": self.navigation_guidance.model_dump() if self.navigation_guidance else None,
            "errors": self.errors,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentState":
        """Create from dictionary."""
        scene = SceneAnalysis(**data["scene_analysis"]) if data.get("scene_analysis") else None
        objects = [DetectedObject(**obj) for obj in data.get("detected_objects", [])]
        guidance = NavigationGuidance(**data["navigation_guidance"]) if data.get("navigation_guidance") else None
        
        return cls(
            instruction=data.get("instruction", ""),
            image_base64=data.get("image_base64"),
            scene_analysis=scene,
            detected_objects=objects,
            navigation_guidance=guidance,
            errors=data.get("errors", []),
        )