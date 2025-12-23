"""Configuration settings for LangGraph workflow."""
import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
FAL_KEY = os.getenv("FAL_KEY")
PRODIA_KEY = os.getenv("PRODIA_KEY")

# Model configurations for content generation tools
IMAGE_MODELS = {
    "fast": {
        "default": "flux-fast-schnell",  # Prodia
    },
    "pro": {
        "default": "fal-ai/flux-krea",
        "photorealistic": "fal-ai/seedream/v4",
        "anime": "fal-ai/flux-krea",
        "artistic": "fal-ai/seedream/v4",
    }
}

VIDEO_MODELS = {
    "fast": "fal-ai/bytedance/seedance/v1/pro/fast/text-to-video",
    "pro": "fal-ai/bytedance/seedance/v1/pro/text-to-video",
}

EDIT_MODEL = "fal-ai/bytedance/seedream/v4/edit"

# Default settings
DEFAULT_IMAGE_WIDTH = 1024
DEFAULT_IMAGE_HEIGHT = 1024
DEFAULT_NUM_IMAGES = 1
DEFAULT_VIDEO_DURATION = 5

# LLM settings (via fal.ai any-llm API)
# Available models: google/gemini-2.5-flash, anthropic/claude-3.5-sonnet,
#                   openai/gpt-4o, meta-llama/llama-3.1-70b-instruct, etc.
DEFAULT_FAL_MODEL_NAME = os.getenv("FAL_MODEL_NAME", "google/gemini-2.5-flash")
DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
DEFAULT_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))
