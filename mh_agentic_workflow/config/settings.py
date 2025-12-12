"""Application settings and configuration"""
import os
from typing import Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings:
    """Application settings loaded from environment variables"""
    
    # API Keys
    FAL_KEY: Optional[str] = os.getenv("FAL_KEY")
    HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")
    
    # Model Configuration
    FAL_MODEL_NAME: str = "google/gemini-2.5-flash"
    MAX_AGENT_STEPS: int = 10
    
    # UI Configuration
    APP_TITLE: str = "Fal.ai Agentic Workflow"
    CHATBOT_HEIGHT: int = 600
    SHARE_LINK: bool = True
    SERVER_NAME: str = "0.0.0.0"
    SERVER_PORT: int = 7860
    
    # Custom CSS
    CUSTOM_CSS: str = """
    .chatbot-container {
        font-family: 'Segoe UI', system-ui, sans-serif;
    }
    """
    
    # Example prompts
    EXAMPLE_PROMPTS = [
        "Generate an image of a futuristic city with flying cars",
        "Create 2 images of a dragon, make them vertical format",
        "Generate a video of ocean waves at sunset, 5 seconds long"
    ]
    
    # Markdown content
    APP_DESCRIPTION: str = """
# 🤖 Fal.ai Agentic Workflow with Smolagents

This agent can:
- 🎨 Generate images using fal.ai models (nano-banana, flux-schnell, flux-pro)
- 🎬 Create videos with text descriptions
- ✏️ Edit and transform uploaded images
- 💭 Stream thinking process and results

**Examples:**
- "Generate 2 images of a sunset over mountains using flux-pro"
- "Create a 5-second video of a cat playing with yarn"
- "Make the image more colorful and vibrant"
"""
    
    AGENT_INFO: str = """
**Available Tools:**
- `fal_image_generation`: Generate images using fal.ai models
- `fal_video_generation`: Create videos from text
- `fal_image_edit`: Edit and transform images

**How it works:**
1. Agent analyzes your request
2. Selects appropriate fal.ai tool and parameters
3. Executes the tool
4. Returns results with images/videos inline

**Supported Models:**
- nano-banana (fast, high quality)
- flux-schnell (fast)
- flux-pro (highest quality)
"""
    
    @classmethod
    def validate(cls) -> None:
        """Validate that required settings are present"""
        if not cls.FAL_KEY:
            raise ValueError("FAL_KEY environment variable is required")
        if not cls.HF_TOKEN:
            raise ValueError("HF_TOKEN environment variable is required")


# Create singleton instance
settings = Settings()
