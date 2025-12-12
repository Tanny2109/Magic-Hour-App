"""Gradio interface components for the application"""
from typing import Optional, List, Tuple, Dict, Any

import gradio as gr
from PIL import Image

from ..agents.legacy_agent import FalAIClient


class GradioInterface:
    """Gradio interface for legacy FalAI client"""
    
    def __init__(self, assistant: Optional[FalAIClient] = None):
        """Initialize the Gradio interface
        
        Args:
            assistant: FalAIClient instance, creates new one if not provided
        """
        self.assistant = assistant if assistant else FalAIClient()
        gr.close_all()  # Close any existing Gradio instances

    def respond(
        self, message: str, history: List
    ) -> Tuple[str, Optional[Image.Image]]:
        """Handle user message and return response
        
        Args:
            message: User input message
            history: Chat history
            
        Yields:
            Tuple of (response_text, image)
        """
        imgs, text = self.assistant.process_message(message, history)
        img0 = imgs[0] if imgs else None
        yield text, img0

    def handle_image_upload(self, image: Optional[Image.Image]) -> str:
        """Handle image uploads
        
        Args:
            image: Uploaded PIL Image
            
        Returns:
            Status message
        """
        if image is not None:
            self.assistant.uploaded_image = image
            return "Image uploaded! You can now ask me to edit, modify, or transform it."
        return ""

    def interact_with_agent(
        self, prompt: str, history: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Handle chat interaction with proper message format
        
        Args:
            prompt: User input
            history: Chat history
            
        Yields:
            Updated chat history
        """
        # Initialize history if None
        history = history or []

        # Add user message
        history.append({"role": "user", "content": prompt})
        yield history

        # Get response from assistant
        imgs, text, video = self.assistant.process_message(prompt, history)

        # Build assistant response content
        content = []
        if text:
            content.append({"type": "text", "text": text})

        # Add images (filter out errors)
        if imgs:
            for img in imgs:
                if img is not None:
                    # Save to temp file for Gradio
                    import tempfile
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                    img.save(temp_file.name)
                    content.append({"path": temp_file.name, "mime_type": "image/png"})
        
        # Add video if present
        if video:
            content.append({"path": video, "mime_type": "video/mp4"})
            

        # Add assistant message to history
        if content:
            history.append({"role": "assistant", "content": content if len(content) > 1 else content[0]})
        else:
            history.append({"role": "assistant", "content": "No response generated."})

        yield history

    def create_chat_interface(self) -> gr.Blocks:
        """Create the Gradio chat interface
        
        Returns:
            Gradio Blocks interface
        """
        custom_css = """
        .chat-container {
            font-family: 'Segoe UI', system-ui, sans-serif;
        }
        .upload-area {
            border: 2px dashed #ccc;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            margin: 10px 0;
        }
        """

        with gr.Blocks(css=custom_css, title="Fal.ai Image Chat Agent") as demo:
            gr.Markdown("""
            # 🎨 Fal.ai Image Generation Agent (Legacy)

            This is the legacy implementation using fal-ai/any-llm for intent detection.

            **Features:**
            - Generate images from text descriptions
            - Uses nano-banana model for high-quality generation
            - Conversation history maintained

            **Note:** For full agentic workflow with tool selection and streaming,
            use the Smolagents version instead.
            """)

            chatbot = gr.Chatbot(
                type="messages",
                label="Agent Chat",
                height=500
            )

            with gr.Row():
                chat_input = gr.Textbox(
                    placeholder="Describe the content you want to generate...",
                    show_label=False,
                    scale=4
                )
                submit_btn = gr.Button("Send", scale=1)

            # Examples
            gr.Examples(
                examples=[
                    "Generate an image of a sunset over mountains",
                    "Create a futuristic cityscape with flying cars",
                    "Make me a picture of a cute cat wearing sunglasses"
                ],
                inputs=chat_input
            )

            # Image upload section
            with gr.Accordion("📤 Upload Image (for editing)", open=False):
                image_upload = gr.Image(
                    type="pil",
                    label="Upload an image to edit"
                )
                upload_status = gr.Textbox(label="Upload Status", interactive=False)

                image_upload.change(
                    self.handle_image_upload,
                    inputs=[image_upload],
                    outputs=[upload_status]
                )

            def handle_submit(prompt: str, history: List):
                """Handle message submission"""
                yield from self.interact_with_agent(prompt, history)

            submit_btn.click(
                handle_submit,
                inputs=[chat_input, chatbot],
                outputs=[chatbot]
            )

            chat_input.submit(
                handle_submit,
                inputs=[chat_input, chatbot],
                outputs=[chatbot]
            )

        return demo
