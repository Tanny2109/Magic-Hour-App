"""Main entry point for Smolagents-based application"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.agents.smolagent_ref import SmolagentFalApp
from config.settings import settings
import gradio as gr


def main():
    """Launch the Smolagents application"""
    # Validate settings
    try:
        settings.validate()
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("Please set FAL_KEY and HF_TOKEN in your .env file")
        sys.exit(1)
    
    # Create and launch app
    print("🚀 Starting Fal.ai Smolagent Application...")
    app = SmolagentFalApp(
        hf_token=settings.HF_TOKEN,
        fal_model_name=settings.FAL_MODEL_NAME
    )
    
    gr.close_all()
    demo = app.create_interface()
    demo.launch(
        share=settings.SHARE_LINK,
        server_name=settings.SERVER_NAME,
        server_port=settings.SERVER_PORT,
        show_error=True,
        debug=True,
    )


if __name__ == "__main__":
    main()
