"""Smolagents implementation for fal.ai workflow"""
import os
import re
from typing import List, Dict, Any, Generator
import fal_client

import gradio as gr
from smolagents import CodeAgent
from ..models import FalAIModel
from dotenv import load_dotenv
load_dotenv()

from ..tools.fal_tools import (
    FalImageGenerationTool,
    FalVideoGenerationTool,
    FalImageEditTool
)
from src.core.utils import stream_from_smolagent
from config.model_config import model_config


class SmolagentFalApp:
    """Smolagents-based application for fal.ai workflow with streaming support"""

    def __init__(self, hf_token: str, fal_model_name="google/gemini-2.5-flash"):
        """Initialize the smolagent application

        Args:
            hf_token: Hugging Face API token
            model_id: LLM model identifier
        """
        # Initialize fal.ai tools
        self.tools = [
            FalImageGenerationTool(),
            FalVideoGenerationTool(),
            FalImageEditTool()
        ]

        # Initialize agent with Hugging Face model
        self.model = FalAIModel(fal_model_name=fal_model_name)

        # Define system prompt for the agent
        self.system_prompt = """You are an intelligent assistant that helps users with image generation, image editing, and video creation using fal.ai models.

Your capabilities:
- Generate images using various models (nano-banana, flux-schnell, flux-pro)
- Create videos from text descriptions
- Edit and transform images based on user requests

When responding:
1. Analyze the user's request carefully
2. Choose the appropriate tool and parameters
3. Execute the task efficiently
4. Provide clear, helpful responses

Always aim to understand user intent and deliver high-quality results."""

        self.agent = CodeAgent(
            tools=self.tools,
            model=self.model,
            max_steps=10,
            # system_prompt=self.system_prompt,
        )

    def _build_context_from_history(self, history: List[Dict[str, Any]]) -> str:
        """Build conversation context from chat history
        
        Args:
            history: List of message dictionaries
            
        Returns:
            Formatted conversation context string
        """
        # Get last few exchanges for context
        recent_history = history if len(history) > 1 else []
        
        context_parts = []
        if recent_history:
            context_parts.append("Previous conversation:")
            for msg in recent_history:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                
                # Skip media-only messages (dict content is for images/videos)
                if isinstance(content, dict):
                    # This is a media file, skip it in text context
                    continue
                elif isinstance(content, list):
                    text_parts = [c.get("text", "") for c in content if isinstance(c, dict) and "text" in c]
                    content = " ".join(text_parts) if text_parts else ""
                    if not content.strip():
                        continue
                
                # Only add if we have actual text content
                if isinstance(content, str) and content.strip():
                    context_parts.append(f"{role.capitalize()}: {content}")
        
        # Add current user message (which should always be text from the user)
        current_msg = history[-1].get("content", "") if history else ""
        if isinstance(current_msg, str) and current_msg.strip():
            context_parts.append(f"\nCurrent request: {current_msg}")
            
        return "\n".join(context_parts)

    def stream_agent_response(
        self, user_message: str, history: List[Dict[str, Any]], agent_prompt: str = None
    ) -> Generator[List[Dict[str, Any]], None, None]:
        """Stream agent thinking and results to Gradio chatbot
        
        Args:
            user_message: User's input message (for display)
            history: Chat history
            agent_prompt: Full prompt with settings (for agent). If None, uses user_message.
            
        Yields:
            Updated chat history with agent responses
        """
        import threading
        import time
        from src.core.utils import parse_image_paths, parse_video_paths

        try:
            # Use agent_prompt if provided, otherwise user_message
            actual_prompt = agent_prompt if agent_prompt is not None else user_message

            # Add user message to history (display version)
            history = history or []
            history.append({"role": "user", "content": user_message})
            yield history
            
            # Build conversation context (using history which now has the clean user message)
            # Note: We might want to inject the settings into the context if they aren't in history.
            # But _build_context_from_history just takes the history.
            # So we should probably append the actual_prompt to the context manually or 
            # temporarily add it to history? 
            # Actually, CodeAgent.run() takes the prompt string. 
            # But here we are passing 'conversation_context' to agent.run().
            # Let's reconstruct the context to include the system settings for this turn.
            
            conversation_context = self._build_context_from_history(history[:-1]) # All history except last
            conversation_context += f"\nCurrent request: {actual_prompt}"
            
            # Add placeholder for assistant response
            history.append({
                "role": "assistant",
                "content": "Analyzing request..."
            })
            yield history

            # Variables to store agent result
            agent_output = None
            agent_error = None
            
            def run_agent():
                nonlocal agent_output, agent_error
                try:
                    agent_output = self.agent.run(conversation_context)
                except Exception as e:
                    agent_error = e

            # Start agent in background thread
            start_time = time.time()
            agent_thread = threading.Thread(target=run_agent)
            agent_thread.start()
            
            # Spinner animation frames
            spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
            spinner_idx = 0
            
            # Loop while agent is running to update UI
            while agent_thread.is_alive():
                elapsed = time.time() - start_time
                spinner = spinner_frames[spinner_idx % len(spinner_frames)]
                spinner_idx += 1
                
                # Update the last message with loading UI
                loading_html = (
                    f'<div style="display: flex; align-items: center; gap: 10px;">'
                    f'<span style="font-size: 1.2em;">{spinner}</span>'
                    f'<span>Generating content...</span>'
                    f'<span style="color: #888; font-family: monospace;">({elapsed:.1f}s)</span>'
                    f'</div>'
                )
                
                history[-1]["content"] = loading_html
                yield history
                
                time.sleep(0.1)
                
            # Wait for thread to ensure we have the result
            agent_thread.join()
            total_time = time.time() - start_time
            
            if agent_error:
                raise agent_error
                
            # Process final result
            output_text = str(agent_output)
            image_paths = parse_image_paths(output_text)
            video_paths = parse_video_paths(output_text)
            
            # Clean text content
            clean_text = output_text.strip()
            
            # DON'T remove the paths - the agent needs them to reference images later!
            # Instead, we'll keep them but format them differently for display
            
            # Remove hallucinated URLs (fal.ai web page links that aren't actual image URLs)
            # These are URLs that don't end in valid image/video extensions
            import re
            # Match URLs but NOT ones ending in valid extensions
            hallucinated_url_pattern = r'https?://[^\s]+(?<!\.png)(?<!\.jpg)(?<!\.jpeg)(?<!\.webp)(?<!\.mp4)(?<!\.mov)(?<!\.avi)(?<!\.webm)'
            clean_text = re.sub(hallucinated_url_pattern, '', clean_text)
            
            # Format the output: wrap file paths in hidden span for better UX
            # but keep them in the text for agent context
            for path in image_paths + video_paths:
                # Replace paths with hidden version (small grey text)
                clean_text = clean_text.replace(
                    path,
                    f'<span style="color: #999; font-size: 0.7em;">{path}</span>'
                )
            
            final_text_content = ""
            
            # Always show the text if we have it (agent needs this context!)
            if clean_text.strip():
                final_text_content = clean_text

            # Add execution time to text
            if final_text_content:
                final_text_content += f"\n<div style='color: #999; font-size: 0.8em; margin-top: 10px; text-align: right;'>Generated in {total_time:.2f}s</div>"
            else:
                final_text_content = f"<div style='color: #999; font-size: 0.8em; margin-top: 10px; text-align: right;'>Generated in {total_time:.2f}s</div>"
            
            # Update the loading message with the text content
            history[-1]["content"] = final_text_content
            
            # Append images as new messages
            for img in image_paths:
                history.append({
                    "role": "assistant",
                    "content": {"path": img}
                })
                
            # Append videos as new messages
            for vid in video_paths:
                history.append({
                    "role": "assistant",
                    "content": {"path": vid}
                })
            
            yield history
                
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}\n\nPlease try rephrasing your request."
            history[-1]["content"] = error_msg
            yield history

    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface
        
        Returns:
            Gradio Blocks interface
        """
        custom_css = """
        .chatbot-container {
            font-family: 'Segoe UI', system-ui, sans-serif;
        }
        """

        with gr.Blocks(css=custom_css, title="Fal.ai Agent") as demo:

            chatbot = gr.Chatbot(
                type="messages",
                label="Agent Chat",
                height=600
            )

            with gr.Row():
                chat_input = gr.MultimodalTextbox(
                    placeholder="Ask me to generate images or videos...",
                    show_label=False,
                    file_count="single",
                    interactive=True
                )

            submit_btn = gr.Button("Send", variant="primary")

            # Performance mode selector
            with gr.Row():
                mode_selector = gr.Radio(
                    label="Performance Mode",
                    choices=["fast", "pro"],
                    value="fast",
                    info="Fast: Uses quickest models. Pro: Uses highest quality models.",
                    interactive=True
                )
            
            # Advanced settings
            with gr.Accordion("⚙️ Advanced Settings", open=False):
                with gr.Row():
                    aspect_ratio = gr.Dropdown(
                        label="Aspect Ratio",
                        choices=["square", "landscape_4_3", "landscape_16_9", "portrait_3_4", "portrait_9_16"],
                        value="square",
                        info="Target aspect ratio for generated images"
                    )
                    num_inference_steps = gr.Slider(
                        label="Inference Steps",
                        minimum=1,
                        maximum=50,
                        value=4,
                        step=1,
                        info="More steps = higher quality but slower"
                    )
                
                with gr.Row():
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        placeholder="blurry, low quality, ugly, deformed...",
                        info="What to avoid in the generation"
                    )
                    seed = gr.Number(
                        label="Seed",
                        value=-1,
                        precision=0,
                        info="Set a specific seed for reproducibility (-1 for random)"
                    )

            # Examples
            gr.Examples(
                examples=[
                    "Generate an image of a futuristic city with flying cars",
                    "Create 2 images of a dragon, make them vertical format",
                    "Generate a video of ocean waves at sunset, 5 seconds long"
                ],
                inputs=chat_input
            )

            def handle_input(
                message: Dict[str, Any], 
                history: List[Dict[str, Any]], 
                mode: str,
                ar: str,
                steps: int,
                neg_prompt: str,
                seed_val: int
            ):
                """Handle both text and file inputs with advanced settings"""
                text_input = message.get("text", "")
                files = message.get("files", [])
                perf_mode = mode or "fast"

                # Construct detailed prompt with settings
                full_prompt = f"{text_input}\n\n[System Settings]"
                full_prompt += f"\nPerformance Mode: {perf_mode}"
                full_prompt += f"\nAspect Ratio: {ar}"
                full_prompt += f"\nInference Steps: {steps}"
                if neg_prompt:
                    full_prompt += f"\nNegative Prompt: {neg_prompt}"
                if seed_val != -1:
                    full_prompt += f"\nSeed: {seed_val}"

                # If file uploaded, add to message
                if files:
                    file_path = files[0] if isinstance(files, list) else files
                    full_prompt += f"\n\nUploaded file: {file_path}"
                    # Also append to text input for display if needed, or just rely on Gradio showing the file
                    # But for the agent prompt, we need it in text.

                # Stream response
                # Pass text_input as user_message (what user sees)
                # Pass full_prompt as agent_prompt (what agent sees)
                for updated_history in self.stream_agent_response(text_input, history, agent_prompt=full_prompt):
                    yield updated_history, None  # Clear input after submission

            # Connect submit and button click
            input_components = [
                chat_input, 
                chatbot, 
                mode_selector,
                aspect_ratio,
                num_inference_steps,
                negative_prompt,
                seed
            ]
            
            submit_events = [
                chat_input.submit(
                    handle_input,
                    inputs=input_components,
                    outputs=[chatbot, chat_input]
                ),
                submit_btn.click(
                    handle_input,
                    inputs=input_components,
                    outputs=[chatbot, chat_input]
                )
            ]
            # Add loading state to button and input
            for event in submit_events:
                event.then(
                    lambda: gr.Button(value="Send", interactive=True),
                    None,
                    submit_btn
                )

        return demo
