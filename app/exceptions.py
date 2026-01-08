"""Custom exceptions for the blind navigation system."""


class BlindNavException(Exception):
    """Base exception for all blind navigation errors."""
    pass


class ModelLoadError(BlindNavException):
    """Raised when a model fails to load."""
    pass


class ImageProcessingError(BlindNavException):
    """Raised when image processing fails."""
    pass


class AgentExecutionError(BlindNavException):
    """Raised when an agent fails during execution."""
    pass


class OrchestratorError(BlindNavException):
    """Raised when the orchestrator encounters an error."""
    pass


class APIError(BlindNavException):
    """Raised when external API calls fail."""
    pass