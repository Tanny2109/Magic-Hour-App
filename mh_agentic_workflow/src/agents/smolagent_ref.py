"""Smolagents implementation for fal.ai workflow"""
from typing import List, Dict, Any
from smolagents import CodeAgent
from dotenv import load_dotenv

from ..models import FalAIModel
from ..tools.fal_tools import (
    FalImageGenerationTool,
    FalVideoGenerationTool,
    FalImageEditTool
)

load_dotenv()


class SmolagentFalApp:
    """Smolagents-based application for fal.ai workflow with streaming support"""

    def __init__(self, hf_token: str, fal_model_name="google/gemini-2.5-flash"):
        self.tools = [
            FalImageGenerationTool(),
            FalVideoGenerationTool(),
            FalImageEditTool()
        ]

        self.model = FalAIModel(fal_model_name=fal_model_name)

        self.agent = CodeAgent(
            tools=self.tools,
            model=self.model,
            max_steps=10,
            planning_interval=3
        )
