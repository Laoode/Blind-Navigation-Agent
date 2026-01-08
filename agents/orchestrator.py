"""Orchestrator agent using Gemini Flash 2.5 for navigation guidance synthesis."""

import logging
from typing import Optional

import google.generativeai as genai

from agents.base import BaseAgent
from app.exceptions import AgentExecutionError, APIError
from models.schemas import AgentState, NavigationGuidance, DetectedObject, SceneAnalysis

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a navigation assistant for blind users. Your role is to provide 
clear, concise, and actionable navigation guidance based on visual scene analysis and 
detected objects.

Guidelines:
1. Prioritize safety - always mention potential hazards first
2. Use clock positions (12 o'clock = straight ahead) for directions
3. Estimate distances in steps or meters
4. Describe obstacles by their position relative to the user's path
5. Keep instructions brief but complete
6. Use consistent, predictable language patterns
7. Focus on the immediate path and obstacles within 5 meters

Response format:
- Start with the most important navigation instruction
- Follow with spatial awareness context
- End with any hazard warnings

Do NOT use emojis or markdown formatting. Speak directly and clearly."""


class OrchestratorAgent(BaseAgent):
    """Orchestrator agent that synthesizes all inputs into navigation guidance.
    
    Uses Gemini Flash 2.5 to:
    - Combine scene analysis and object detections
    - Generate natural language navigation instructions
    - Prioritize safety-critical information
    """
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash-preview-05-20"):
        super().__init__(name="orchestrator")
        self._api_key = api_key
        self._model_name = model_name
        self._model = None
    
    async def initialize(self) -> None:
        """Initialize Gemini client."""
        try:
            genai.configure(api_key=self._api_key)
            self._model = genai.GenerativeModel(
                model_name=self._model_name,
                system_instruction=SYSTEM_PROMPT,
            )
            logger.info(f"Orchestrator initialized with model: {self._model_name}")
            self._initialized = True
            
        except Exception as e:
            raise APIError(f"Failed to initialize Gemini: {e}") from e
    
    async def process(self, state: AgentState) -> AgentState:
        """Generate navigation guidance from aggregated context.
        
        Args:
            state: State with scene_analysis and detected_objects.
            
        Returns:
            State updated with navigation_guidance.
        """
        if not self._initialized:
            state.errors.append("Orchestrator not initialized")
            return state
        
        try:
            guidance = await self._generate_guidance(state)
            state.navigation_guidance = guidance
            logger.debug(f"Guidance generated: {guidance.primary_instruction[:100]}...")
            
        except Exception as e:
            error_msg = f"Guidance generation failed: {e}"
            logger.error(error_msg)
            state.errors.append(error_msg)
            state.navigation_guidance = self._fallback_guidance(state)
        
        return state
    
    async def _generate_guidance(self, state: AgentState) -> NavigationGuidance:
        """Generate navigation guidance using Gemini.
        
        Args:
            state: Current workflow state.
            
        Returns:
            NavigationGuidance with instructions.
        """
        prompt = self._build_prompt(state)
        
        try:
            response = await self._model.generate_content_async(
                prompt,
                generation_config=genai.GenerationConfig(
                    max_output_tokens=256,
                    temperature=0.3,
                ),
            )
            
            guidance_text = response.text.strip()
            
            return self._parse_guidance(guidance_text, state)
            
        except Exception as e:
            raise APIError(f"Gemini API call failed: {e}") from e
    
    def _build_prompt(self, state: AgentState) -> str:
        """Build prompt from state context.
        
        Args:
            state: Current state with analysis results.
            
        Returns:
            Formatted prompt string.
        """
        parts = [f"User query: {state.instruction}\n"]
        
        if state.scene_analysis:
            parts.append("Scene Analysis:")
            parts.append(f"- Description: {state.scene_analysis.description}")
            parts.append(f"- Spatial context: {state.scene_analysis.spatial_context}")
            if state.scene_analysis.potential_hazards:
                hazards = ", ".join(state.scene_analysis.potential_hazards)
                parts.append(f"- Potential hazards: {hazards}")
            parts.append("")
        
        if state.detected_objects:
            parts.append(f"Detected Objects ({len(state.detected_objects)} items):")
            for obj in state.detected_objects[:7]:
                parts.append(
                    f"- {obj.label} ({obj.confidence:.0%} confidence) - "
                    f"Position: {obj.position_description}"
                )
            parts.append("")
        
        parts.append(
            "Based on this information, provide navigation guidance for a blind user. "
            "Be specific about positions and distances."
        )
        
        return "\n".join(parts)
    
    def _parse_guidance(self, text: str, state: AgentState) -> NavigationGuidance:
        """Parse Gemini response into structured guidance.
        
        Args:
            text: Raw response text.
            state: Current state for context.
            
        Returns:
            Structured NavigationGuidance.
        """
        lines = text.strip().split("\n")
        
        primary = lines[0] if lines else "Unable to generate guidance."
        spatial = ""
        hazards = []
        
        for line in lines[1:]:
            line_lower = line.lower()
            if any(word in line_lower for word in ["warning", "caution", "careful", "hazard", "danger"]):
                hazards.append(line.strip("- "))
            elif spatial == "" and line.strip():
                spatial = line.strip("- ")
        
        confidence = "high"
        if state.errors:
            confidence = "low"
        elif not state.scene_analysis or not state.detected_objects:
            confidence = "medium"
        
        return NavigationGuidance(
            primary_instruction=primary,
            spatial_awareness=spatial,
            hazard_warnings=hazards,
            confidence_level=confidence,
        )
    
    def _fallback_guidance(self, state: AgentState) -> NavigationGuidance:
        """Generate fallback guidance when API fails.
        
        Args:
            state: Current state.
            
        Returns:
            Basic safety guidance.
        """
        objects_info = ""
        if state.detected_objects:
            obj_names = [obj.label for obj in state.detected_objects[:3]]
            objects_info = f" Detected nearby: {', '.join(obj_names)}."
        
        return NavigationGuidance(
            primary_instruction=f"Proceed with caution.{objects_info}",
            spatial_awareness="Unable to fully analyze scene.",
            hazard_warnings=["System operating in fallback mode - exercise extra caution"],
            confidence_level="low",
        )