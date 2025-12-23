"""Smolagents implementation for fal.ai workflow - Production version with vision-first context"""
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
    """Smolagents-based application for fal.ai workflow with vision-first context understanding"""

    def __init__(self, hf_token: str, fal_model_name="google/gemini-2.5-flash"):
        self.tools = [
            FalImageGenerationTool(),
            FalVideoGenerationTool(),
            FalImageEditTool()
        ]

        self.model = FalAIModel(fal_model_name=fal_model_name)

        # Production system prompt: Works with pre-analyzed visual context
        system_prompt = """You are an expert AI image/video generation assistant powered by fal.ai.

## VISUAL CONTEXT INTEGRATION

When you receive a [VISUAL CONTEXT ANALYSIS] section in the prompt, this contains an analysis of previous images from the conversation performed by a vision model. USE THIS ANALYSIS to:

1. **Understand the theme/universe** - If the analysis identifies characters from Mortal Kombat, Marvel, anime, etc., interpret user requests in that context
2. **Maintain consistency** - Use the identified art style, setting, and theme when editing or generating follow-ups
3. **Disambiguate references** - If the analysis says "scorpio likely refers to Scorpion character", trust that interpretation
4. **Choose the right tool** - Follow the analysis's recommendation on whether to EDIT or GENERATE NEW

## TOOL SELECTION

- `fal_image_generation`: Create NEW images from scratch
- `fal_image_edit`: Modify EXISTING images (requires file paths from [Image File Paths] section)
- `fal_video_generation`: Create videos from text

When the visual analysis recommends editing → use `fal_image_edit` with the provided file paths
When the user wants something completely new → use `fal_image_generation`

## REASONING GUIDELINES

- Explain WHY you're choosing specific tools and parameters
- Reference the visual analysis when making decisions
- If the analysis identified a specific character/universe, mention it in your reasoning
- For batch edits, use comma-separated paths for parallel processing

## EXAMPLES

✓ "The visual analysis identified Sub-Zero from Mortal Kombat. The user's request to 'add scorpio' means adding the Scorpion character. I'll use fal_image_edit with the previous image path to add Scorpion in the same style."

✓ "Analysis shows anime-style images. User wants 'a cat' - I'll generate/edit to include an anime-style cat to maintain consistency."

✓ "The user wants to edit all 4 images with the same prompt, so I'll batch them in a single fal_image_edit call with comma-separated paths for faster processing."

Be conversational but thorough. Help users understand your reasoning."""

        self.agent = CodeAgent(
            tools=self.tools,
            model=self.model,
            max_steps=10,
            planning_interval=3,  # More frequent planning for better context use
            # system_prompt=system_prompt #this is not a valid arg for CodeAgent
        )
