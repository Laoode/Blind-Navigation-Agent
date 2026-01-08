"""Base agent interface defining the contract for all specialized agents."""

from abc import ABC, abstractmethod
from typing import Any

from models.schemas import AgentState


class BaseAgent(ABC):
    """Abstract base class for all navigation agents.
    
    Each agent implements a specific capability in the multi-agent system:
    - Scene analysis (FastVLM)
    - Object detection (YOLOv8)
    - Orchestration (Gemini)
    """
    
    def __init__(self, name: str):
        self._name = name
        self._initialized = False
    
    @property
    def name(self) -> str:
        """Agent identifier for logging and debugging."""
        return self._name
    
    @property
    def is_initialized(self) -> bool:
        """Check if agent is ready to process."""
        return self._initialized
    
    @abstractmethod
    async def initialize(self) -> None:
        """Load models and prepare for inference.
        
        Raises:
            ModelLoadError: If model loading fails.
        """
        pass
    
    @abstractmethod
    async def process(self, state: AgentState) -> AgentState:
        """Process the current state and return updated state.
        
        Args:
            state: Current workflow state containing image and context.
            
        Returns:
            Updated state with this agent's contributions.
            
        Raises:
            AgentExecutionError: If processing fails.
        """
        pass
    
    async def cleanup(self) -> None:
        """Release resources. Override if agent holds resources."""
        self._initialized = False
    
    def __repr__(self) -> str:
        status = "initialized" if self._initialized else "not initialized"
        return f"{self.__class__.__name__}(name={self._name}, status={status})"