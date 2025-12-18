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

        # Enhanced system prompt for more detailed reasoning
        enhanced_system_prompt = """You are an expert AI image/video generation assistant powered by fal.ai.

REASONING GUIDELINES - Be Detailed and Transparent:
- When planning, explain WHY you're choosing specific tools and parameters
- Describe what you're trying to achieve with each step
- If selecting model types (fast vs pro), explain your reasoning
- When adjusting parameters (size, aspect ratio, etc.), explain the creative intent
- After generation, briefly note what you observe about the results

EXAMPLES OF GOOD REASONING:
✓ "I'll use flux-pro model for this request because the user wants high artistic quality. Setting aspect ratio to 16:9 for a cinematic composition."
✓ "Generating 4 variations to give the user options. Each will have slightly different interpretations of 'sunset' - from warm orange tones to cooler purple hues."
✓ "The user wants to edit all 4 images with the same prompt, so I'll batch them in a single call for faster processing (parallel execution)."

Be conversational but thorough in your thought process. Help users understand how AI image generation works."""

        self.agent = CodeAgent(
            tools=self.tools,
            model=self.model,
            max_steps=10,
            planning_interval=6,
        )
