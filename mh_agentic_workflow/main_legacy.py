"""Main entry point for legacy application"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.agents.legacy_agent import FalAIClient
from src.ui.gradio_interface import GradioInterface
from config.settings import settings


def main():
    """Launch the legacy application"""
    # Validate settings
    try:
        settings.validate()
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("Please set FAL_KEY and HF_TOKEN in your .env file")
        sys.exit(1)
    
    # Create and launch app
    print("🚀 Starting Fal.ai Legacy Application...")
    assistant = FalAIClient()
    gradio_app = GradioInterface(assistant)
    demo = gradio_app.create_chat_interface()

    demo.launch(
        share=settings.SHARE_LINK,
        server_name=settings.SERVER_NAME,
        server_port=settings.SERVER_PORT,
        show_error=True,
        debug=True,
    )


if __name__ == "__main__":
    main()
