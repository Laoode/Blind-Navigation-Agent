"""LangGraph workflow for multi-agent navigation system.

This module defines the graph structure that orchestrates the three agents:
1. Scene Analyzer (FastVLM) - Visual scene understanding
2. Object Detector (YOLOv8) - Object detection and localization
3. Orchestrator (Gemini) - Navigation guidance synthesis

The workflow runs scene analysis and object detection in parallel,
then combines results in the orchestrator for final guidance.
"""

import asyncio
import logging
import time
from typing import Any, TypedDict

from langgraph.graph import StateGraph, END

from agents.object_detector import ObjectDetectorAgent
from agents.orchestrator import OrchestratorAgent
from agents.scene_analyzer import SceneAnalyzerAgent
from app.config import Settings
from models.schemas import (
    AgentState,
    DetectedObject,
    NavigationGuidance,
    ProcessingRequest,
    ProcessingResponse,
    SceneAnalysis,
)

logger = logging.getLogger(__name__)


class GraphState(TypedDict):
    """State schema for LangGraph workflow."""
    instruction: str
    image_base64: str | None
    scene_analysis: dict | None
    detected_objects: list[dict]
    navigation_guidance: dict | None
    errors: list[str]


class NavigationWorkflow:
    """Manages the multi-agent navigation workflow.
    
    Workflow structure:
    
        [Input]
           |
           v
      +---------+
      | Parallel |
      +---------+
       /       \
      v         v
    [Scene]  [Objects]
      \         /
       v       v
      +---------+
      | Combine |
      +---------+
           |
           v
    [Orchestrator]
           |
           v
       [Output]
    """
    
    def __init__(self, settings: Settings):
        self._settings = settings
        self._scene_analyzer: SceneAnalyzerAgent | None = None
        self._object_detector: ObjectDetectorAgent | None = None
        self._orchestrator: OrchestratorAgent | None = None
        self._graph: StateGraph | None = None
        self._compiled_graph = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize all agents and compile the workflow graph."""
        logger.info("Initializing navigation workflow...")
        
        self._scene_analyzer = SceneAnalyzerAgent(
            model_path=self._settings.fastvlm_model_path
        )
        self._object_detector = ObjectDetectorAgent(
            model_path=self._settings.yolo_model_path
        )
        self._orchestrator = OrchestratorAgent(
            api_key=self._settings.gemini_api_key,
            model_name=self._settings.gemini_model,
        )
        
        await asyncio.gather(
            self._scene_analyzer.initialize(),
            self._object_detector.initialize(),
            self._orchestrator.initialize(),
        )
        
        self._build_graph()
        self._initialized = True
        logger.info("Navigation workflow initialized successfully")
    
    def _build_graph(self) -> None:
        """Construct the LangGraph workflow."""
        workflow = StateGraph(GraphState)
        
        workflow.add_node("analyze_scene", self._analyze_scene_node)
        workflow.add_node("detect_objects", self._detect_objects_node)
        workflow.add_node("combine_results", self._combine_results_node)
        workflow.add_node("generate_guidance", self._generate_guidance_node)
        
        workflow.set_entry_point("analyze_scene")
        
        # Scene analysis leads to object detection (sequential for MVP)
        # In production, these can run in parallel using branching
        workflow.add_edge("analyze_scene", "detect_objects")
        workflow.add_edge("detect_objects", "combine_results")
        workflow.add_edge("combine_results", "generate_guidance")
        workflow.add_edge("generate_guidance", END)
        
        self._compiled_graph = workflow.compile()
    
    async def _analyze_scene_node(self, state: GraphState) -> GraphState:
        """Node: Run scene analysis."""
        agent_state = self._state_from_graph(state)
        result = await self._scene_analyzer.process(agent_state)
        
        return {
            **state,
            "scene_analysis": result.scene_analysis.model_dump() if result.scene_analysis else None,
            "errors": result.errors,
        }
    
    async def _detect_objects_node(self, state: GraphState) -> GraphState:
        """Node: Run object detection."""
        agent_state = self._state_from_graph(state)
        result = await self._object_detector.process(agent_state)
        
        return {
            **state,
            "detected_objects": [obj.model_dump() for obj in result.detected_objects],
            "errors": state.get("errors", []) + result.errors,
        }
    
    async def _combine_results_node(self, state: GraphState) -> GraphState:
        """Node: Combine scene analysis and object detection results."""
        # This node serves as a sync point before orchestration
        # In production, add additional processing logic here
        return state
    
    async def _generate_guidance_node(self, state: GraphState) -> GraphState:
        """Node: Generate final navigation guidance."""
        agent_state = self._state_from_graph(state)
        result = await self._orchestrator.process(agent_state)
        
        return {
            **state,
            "navigation_guidance": result.navigation_guidance.model_dump() if result.navigation_guidance else None,
            "errors": state.get("errors", []) + result.errors,
        }
    
    def _state_from_graph(self, graph_state: GraphState) -> AgentState:
        """Convert GraphState to AgentState for agent processing."""
        scene = None
        if graph_state.get("scene_analysis"):
            scene = SceneAnalysis(**graph_state["scene_analysis"])
        
        objects = [
            DetectedObject(**obj) 
            for obj in graph_state.get("detected_objects", [])
        ]
        
        guidance = None
        if graph_state.get("navigation_guidance"):
            guidance = NavigationGuidance(**graph_state["navigation_guidance"])
        
        return AgentState(
            instruction=graph_state.get("instruction", ""),
            image_base64=graph_state.get("image_base64"),
            scene_analysis=scene,
            detected_objects=objects,
            navigation_guidance=guidance,
            errors=list(graph_state.get("errors", [])),
        )
    
    async def process(self, request: ProcessingRequest) -> ProcessingResponse:
        """Process a navigation request through the workflow.
        
        Args:
            request: Input containing instruction and optional image.
            
        Returns:
            ProcessingResponse with navigation guidance.
        """
        if not self._initialized:
            return ProcessingResponse(
                guidance="System not initialized",
                error="Workflow not initialized",
            )
        
        start_time = time.perf_counter()
        
        initial_state: GraphState = {
            "instruction": request.instruction,
            "image_base64": request.image_base64,
            "scene_analysis": None,
            "detected_objects": [],
            "navigation_guidance": None,
            "errors": [],
        }
        
        try:
            final_state = await self._compiled_graph.ainvoke(initial_state)
            processing_time = (time.perf_counter() - start_time) * 1000
            
            guidance = final_state.get("navigation_guidance", {})
            scene = final_state.get("scene_analysis", {})
            objects = final_state.get("detected_objects", [])
            errors = final_state.get("errors", [])
            
            guidance_text = guidance.get("primary_instruction", "No guidance available")
            if guidance.get("spatial_awareness"):
                guidance_text += f" {guidance['spatial_awareness']}"
            if guidance.get("hazard_warnings"):
                warnings = "; ".join(guidance["hazard_warnings"])
                guidance_text += f" Warning: {warnings}"
            
            return ProcessingResponse(
                guidance=guidance_text,
                scene_description=scene.get("description", "") if scene else "",
                objects_detected=len(objects),
                processing_time_ms=processing_time,
                error="; ".join(errors) if errors else None,
            )
            
        except Exception as e:
            logger.exception("Workflow execution failed")
            processing_time = (time.perf_counter() - start_time) * 1000
            return ProcessingResponse(
                guidance="Unable to process request",
                processing_time_ms=processing_time,
                error=str(e),
            )
    
    async def cleanup(self) -> None:
        """Release all agent resources."""
        cleanup_tasks = []
        if self._scene_analyzer:
            cleanup_tasks.append(self._scene_analyzer.cleanup())
        if self._object_detector:
            cleanup_tasks.append(self._object_detector.cleanup())
        if self._orchestrator:
            cleanup_tasks.append(self._orchestrator.cleanup())
        
        await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        self._initialized = False
        logger.info("Navigation workflow cleaned up")