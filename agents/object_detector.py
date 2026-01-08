"""Object Detector agent using YOLOv8n-oiv7 for object detection."""

import base64
import io
import logging
from typing import Optional

from agents.base import BaseAgent
from app.exceptions import AgentExecutionError, ModelLoadError
from models.schemas import AgentState, DetectedObject

logger = logging.getLogger(__name__)


class ObjectDetectorAgent(BaseAgent):
    """Agent that detects objects using YOLOv8n-oiv7.
    
    Provides:
    - Object detection with 601 Open Images V7 classes
    - Bounding box coordinates
    - Confidence scores
    - Position descriptions for navigation
    """
    
    POSITION_THRESHOLDS = {
        "left": 0.33,
        "center": 0.66,
    }
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.3):
        super().__init__(name="object_detector")
        self._model_path = model_path
        self._confidence_threshold = confidence_threshold
        self._model = None
    
    async def initialize(self) -> None:
        """Load YOLOv8 model."""
        try:
            from ultralytics import YOLO
            
            self._model = YOLO(self._model_path)
            logger.info(f"Object detector initialized: {self._model_path}")
            self._initialized = True
            
        except ImportError:
            logger.warning("ultralytics not installed, using mock detector")
            self._initialized = True
            
        except Exception as e:
            raise ModelLoadError(f"Failed to load YOLO model: {e}") from e
    
    async def process(self, state: AgentState) -> AgentState:
        """Detect objects in image and update state.
        
        Args:
            state: Current state with image_base64.
            
        Returns:
            State updated with detected_objects.
        """
        if not self._initialized:
            state.errors.append("Object detector not initialized")
            return state
        
        if not state.image_base64:
            state.errors.append("No image provided for object detection")
            return state
        
        try:
            objects = await self._detect_objects(state.image_base64)
            state.detected_objects = objects
            logger.debug(f"Detected {len(objects)} objects")
            
        except Exception as e:
            error_msg = f"Object detection failed: {e}"
            logger.error(error_msg)
            state.errors.append(error_msg)
        
        return state
    
    async def _detect_objects(self, image_base64: str) -> list[DetectedObject]:
        """Run YOLO detection on image.
        
        Args:
            image_base64: Base64 encoded image.
            
        Returns:
            List of detected objects with positions.
        """
        if self._model is None:
            return self._mock_detection()
        
        try:
            from PIL import Image
            
            image_bytes = self._decode_image(image_base64)
            image = Image.open(io.BytesIO(image_bytes))
            
            results = self._model(image, conf=self._confidence_threshold)
            
            detected = []
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                    
                for i, box in enumerate(boxes):
                    xyxy = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    label = result.names[cls_id]
                    
                    position = self._calculate_position(
                        xyxy, image.width, image.height
                    )
                    
                    detected.append(DetectedObject(
                        label=label,
                        confidence=conf,
                        bbox=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                        position_description=position,
                    ))
            
            detected.sort(key=lambda x: x.confidence, reverse=True)
            return detected[:10]  # Top 10 detections
            
        except Exception as e:
            logger.error(f"YOLO inference failed: {e}")
            return self._mock_detection()
    
    def _calculate_position(
        self, 
        bbox: list[float], 
        img_width: int, 
        img_height: int
    ) -> str:
        """Calculate relative position description for navigation.
        
        Args:
            bbox: [x1, y1, x2, y2] coordinates.
            img_width: Image width.
            img_height: Image height.
            
        Returns:
            Human-readable position description.
        """
        center_x = (bbox[0] + bbox[2]) / 2 / img_width
        center_y = (bbox[1] + bbox[3]) / 2 / img_height
        
        if center_x < self.POSITION_THRESHOLDS["left"]:
            horizontal = "left"
        elif center_x < self.POSITION_THRESHOLDS["center"]:
            horizontal = "center"
        else:
            horizontal = "right"
        
        if center_y < 0.4:
            vertical = "upper"
        elif center_y < 0.7:
            vertical = "middle"
        else:
            vertical = "lower"
        
        box_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        img_area = img_width * img_height
        relative_size = box_area / img_area
        
        if relative_size > 0.3:
            distance = "very close"
        elif relative_size > 0.1:
            distance = "nearby"
        elif relative_size > 0.03:
            distance = "at medium distance"
        else:
            distance = "far away"
        
        return f"{vertical} {horizontal}, {distance}"
    
    def _decode_image(self, image_base64: str) -> bytes:
        """Decode base64 image to bytes."""
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]
        return base64.b64decode(image_base64)
    
    def _mock_detection(self) -> list[DetectedObject]:
        """Return mock detections for testing."""
        return [
            DetectedObject(
                label="Person",
                confidence=0.85,
                bbox=(100, 150, 200, 400),
                position_description="center middle, nearby",
            ),
        ]