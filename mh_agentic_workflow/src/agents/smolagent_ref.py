"""Smolagents implementation for fal.ai workflow with Real-Time Streaming"""
import os
import re
from typing import List, Dict, Any, Generator
import gradio as gr
from smolagents import CodeAgent, ActionStep
from dotenv import load_dotenv

# Ensure these imports match your project structure
from ..models import FalAIModel
from ..tools.fal_tools import (
    FalImageGenerationTool,
    FalVideoGenerationTool,
    FalImageEditTool
)
# We don't need stream_from_smolagent anymore as we implement custom streaming here

load_dotenv()

class SmolagentFalApp:
    """Smolagents-based application for fal.ai workflow with TRUE streaming support"""

    def __init__(self, hf_token: str, fal_model_name="google/gemini-2.5-flash"):
        # Initialize fal.ai tools
        self.tools = [
            FalImageGenerationTool(),
            FalVideoGenerationTool(),
            FalImageEditTool()
        ]

        # Initialize agent with Fal/LLM model
        self.model = FalAIModel(fal_model_name=fal_model_name)

        self.agent = CodeAgent(
            tools=self.tools,
            model=self.model,
            max_steps=10,
            add_base_tools=True, # Helpful for basic logic
            additional_authorized_imports=["fal_client", "json"]
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

    def _process_step_for_ui(self, step_log: ActionStep) -> Generator[gr.ChatMessage, None, None]:
        """
        Converts a raw agent step into a 'Magical' UI bubble.
        """
        if isinstance(step_log, ActionStep):
            # 1. VISUALIZE THOUGHTS (The Code)
            code_content = None
            if hasattr(step_log, "tool_calls") and step_log.tool_calls:
                tool_call = step_log.tool_calls[0]
                # Use the tool name directly from the call for clarity
                tool_name = tool_call.name
                code_content = f"```python\n# 🤖 Agent is coding...\n{tool_call.arguments}\n```"
            else:
                tool_name = "Logic"

            # Yield the 'Thinking' bubble
            yield gr.ChatMessage(
                role="assistant",
                content=code_content or "Thinking..."
            )

            # 2. VISUALIZE RESULTS (The Media)
            if step_log.observations:
                observation = str(step_log.observations).strip()
                
                # Check for Media Paths
                is_video = any(ext in observation for ext in [".mp4", ".mov", ".avi", ".webm"])
                is_image = any(ext in observation for ext in [".png", ".jpg", ".webp", ".jpeg"])
                
                # Check if it looks like a path (/tmp/...) or a URL (http://...)
                if (is_video or is_image) and ("/" in observation or "http" in observation):
                    # Clean up the path. Your tool returns "Generated image(s): /path/to/img.png"
                    # We need to extract just the path.
                    import re
                    # Find the first thing that looks like a path or URL
                    match = re.search(r'(/(?:[^/]+/)*[^/]+?\.(?:png|jpg|jpeg|webp|mp4|mov|avi|webm)|https?://\S+)', observation)
                    if match:
                        media_path = match.group(1)

                        if is_video:
                            yield gr.ChatMessage(
                                role="assistant",
                                content={"path": media_path}, 
                                metadata={"title": "Video Generated", "status": "done"}
                            )
                        else:
                            yield gr.ChatMessage(
                                role="assistant",
                                content={"path": media_path},
                                metadata={"title": "Image Generated", "status": "done"}
                            )
                    else:
                        # Fallback if regex fails but it looked like media
                        yield gr.ChatMessage(
                            role="assistant",
                            content={"path": observation},
                            metadata={"title": "Done", "status": "done"}
                        )
                else:
                    yield gr.ChatMessage(
                        role="assistant",
                        content=f"Observation: {observation}",
                        metadata={"title": "Execution Result", "status": "done"}
                    )
                    
    def stream_agent_response(
        self, user_message: str, history: List[Dict[str, Any]], agent_prompt: str = None
    ) -> Generator[List[Dict[str, Any]], None, None]:
        """
        True streaming generator. No threads. No fake spinners.
        """
        # 1. Setup UI History
        actual_prompt = agent_prompt if agent_prompt is not None else user_message
        conversation_context = self._build_context_from_history(history[:-1]) # All history except last
        conversation_context += f"\nCurrent request: {actual_prompt}"
        # We need to rebuild the list of ChatMessages for Gradio
        messages = history if history else []
        messages.append({"role": "user", "content": user_message})
        yield messages

        # 2. Prepare Prompt
        actual_prompt = agent_prompt if agent_prompt is not None else user_message
        
        # We append previous history context manually if needed, 
        # but CodeAgent usually handles its own memory if the instance persists.
        # If this is a one-off run, we pass context.
        # For simplicity in this 'magical' version, we trust the agent's run loop.

        try:
            # 3. THE MAGIC LOOP (stream=True)
            # This yields steps AS THEY HAPPEN.
            for step in self.agent.run(conversation_context, stream=True):
                
                # Convert the internal step to a UI message
                new_ui_messages = list(self._process_step_for_ui(step))
                
                for msg in new_ui_messages:
                    messages.append(msg)
                    # Yield immediately to update UI
                    yield messages
            
            # 4. Final Success Marker (Optional)
            # You can check if the last message was a success
            pass 

        except Exception as e:
            messages.append(gr.ChatMessage(role="assistant", content=f"❌ Error: {str(e)}"))
            yield messages

    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface"""
        custom_css = """
        .chatbot-container { font-family: 'Segoe UI', system-ui, sans-serif; }
        """

        with gr.Blocks(css=custom_css, title="Fal.ai Agent") as demo:

            # TYPE="MESSAGES" IS CRITICAL for the bubbles to work
            chatbot = gr.Chatbot(
                type="messages", 
                label="Agent Chat",
                height=600,
                avatar_images=(None, "https://avatars.githubusercontent.com/u/58669933?v=4")
            )

            with gr.Row():
                chat_input = gr.MultimodalTextbox(
                    placeholder="Ask me to generate images or videos...",
                    show_label=False,
                    file_count="single",
                    interactive=True
                )

            submit_btn = gr.Button("Send", variant="primary")

            # --- Settings (Kept your existing structure) ---
            with gr.Row():
                mode_selector = gr.Radio(
                    label="Performance Mode",
                    choices=["fast", "pro"],
                    value="fast",
                    info="Fast: Uses quickest models. Pro: Uses highest quality models.",
                    interactive=True
                )
            
            with gr.Accordion("⚙️ Advanced Settings", open=False):
                with gr.Row():
                    aspect_ratio = gr.Dropdown(
                        label="Aspect Ratio",
                        choices=["square", "landscape_4_3", "landscape_16_9", "portrait_3_4", "portrait_9_16"],
                        value="square"
                    )
                    num_inference_steps = gr.Slider(
                        label="Inference Steps", minimum=1, maximum=50, value=4, step=1
                    )
                
                with gr.Row():
                    negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="blurry...")
                    seed = gr.Number(label="Seed", value=-1, precision=0)

            # --- Event Handling ---
            def handle_input(message, history, mode, ar, steps, neg_prompt, seed_val):
                text_input = message.get("text", "")
                files = message.get("files", [])
                
                # Construct System Settings Injection
                full_prompt = f"{text_input}\n\n[User Technical Constraints]"
                full_prompt += f"\n- Performance Mode: {mode}"
                full_prompt += f"\n- Aspect Ratio: {ar}"
                full_prompt += f"\n- Inference Steps: {steps}"
                if neg_prompt: full_prompt += f"\n- Negative Prompt: {neg_prompt}"
                if seed_val != -1: full_prompt += f"\n- Seed: {seed_val}"
                
                if files:
                    full_prompt += f"\n\nUploaded file path: {files[0]}"

                # Call the streaming generator
                # Note: We pass text_input for display, but full_prompt for logic
                for updated_history in self.stream_agent_response(text_input, history, agent_prompt=full_prompt):
                    yield updated_history, None

            # Connect inputs
            input_components = [chat_input, chatbot, mode_selector, aspect_ratio, num_inference_steps, negative_prompt, seed]
            
            chat_input.submit(handle_input, input_components, [chatbot, chat_input])
            submit_btn.click(handle_input, input_components, [chatbot, chat_input])

        return demo