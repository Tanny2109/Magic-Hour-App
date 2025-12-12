"""Agent implementations for fal.ai workflow"""

from .smolagent import SmolagentFalApp
from .legacy_agent import FalAIClient

__all__ = ["SmolagentFalApp", "FalAIClient"]
