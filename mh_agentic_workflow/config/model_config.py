"""Model configuration for different performance modes"""
import json
from typing import Literal


class ModelConfig:
    """Configuration for model selection based on performance mode"""

    # Define model mappings for different modes
    MODELS = {
        "fast": {
            "image_generation": "fal-ai/gemini-25-flash-image",  # Gemini 2.5 Flash (Nano-Banana) - fastest
            "video_generation": "fal-ai/veo3/fast",  # Veo3 fast mode
            "image_edit": "fal-ai/nano-banana/edit",  # Always use nano-banana-edit
        },
        "pro": {
            "image_generation": "fal-ai/flux-pro/v1.1-ultra",  # Highest quality
            "video_generation": "fal-ai/veo3",  # Veo3 regular mode (higher quality)
            "image_edit": "fal-ai/nano-banana/edit",  # Always use nano-banana-edit
        }
    }

    def __init__(self, mode: Literal["fast", "pro"] = "fast"):
        """Initialize model configuration

        Args:
            mode: Performance mode - 'fast' or 'pro'
        """
        self.mode = mode
        self._config = self.MODELS[mode]

    def get_model(self, task: Literal["image_generation", "video_generation", "image_edit"]) -> str:
        """Get the model for a specific task

        Args:
            task: The task type

        Returns:
            The model API endpoint for the task
        """
        return self._config[task]

    def set_mode(self, mode: Literal["fast", "pro"]) -> None:
        """Change the performance mode

        Args:
            mode: The new performance mode
        """
        self.mode = mode
        self._config = self.MODELS[mode]

    def to_json(self) -> str:
        """Export configuration as JSON for debugging

        Returns:
            JSON string representation of current config
        """
        config_dict = {
            "mode": self.mode,
            "models": self._config
        }
        return json.dumps(config_dict, indent=2)

    def __repr__(self) -> str:
        return f"ModelConfig(mode='{self.mode}')"


# Create global config instance
model_config = ModelConfig(mode="fast")
