"""
FastAPI backend for Magic Hour AI - streams agent reasoning and results via SSE
"""
import os
import sys
import json
import asyncio
import threading
import time
import re
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add the mh_agentic_workflow to path
WORKFLOW_PATH = Path(__file__).parent.parent / "mh_agentic_workflow"
sys.path.insert(0, str(WORKFLOW_PATH))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(WORKFLOW_PATH / ".env")

# Import agent components (from the existing workflow)
from src.agents.smolagent_ref import SmolagentFalApp
from src.core.utils import parse_image_paths, parse_video_paths
from config.settings import settings

# Initialize the agent
agent_app: Optional[SmolagentFalApp] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize agent on startup"""
    global agent_app
    print("🚀 Initializing Magic Hour AI Agent...")
    try:
        settings.validate()
        agent_app = SmolagentFalApp(
            hf_token=settings.HF_TOKEN,
            fal_model_name=settings.FAL_MODEL_NAME
        )
        print("✅ Agent initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        raise
    yield
    print("👋 Shutting down...")

app = FastAPI(title="Magic Hour AI", lifespan=lifespan)

# Detect production mode
PRODUCTION_MODE = os.getenv("PRODUCTION", "false").lower() == "true"

# CORS for React frontend
allowed_origins = ["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"]
if PRODUCTION_MODE:
    # Add your production domain here
    production_domain = os.getenv("FRONTEND_URL", "*")
    if production_domain != "*":
        allowed_origins.append(production_domain)
    else:
        allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Serve static frontend in production
if PRODUCTION_MODE:
    frontend_dist = Path(__file__).parent / "dist"
    if frontend_dist.exists():
        app.mount("/assets", StaticFiles(directory=str(frontend_dist / "assets")), name="assets")
        print(f"✅ Serving frontend from {frontend_dist}")


class ChatRequest(BaseModel):
    message: str
    settings: dict = {}
    history: list = []  # Conversation history with image paths


class SSEEvent:
    """Helper to format SSE events"""
    @staticmethod
    def format(event_type: str, data: dict) -> str:
        return f"data: {json.dumps({'type': event_type, **data})}\n\n"


def create_blur_thumbnail(image_path: str, size: int = 20) -> str:
    """Create tiny base64 thumbnail for blur effect (ChatGPT-style)"""
    try:
        from PIL import Image
        import io
        import base64

        img = Image.open(image_path)
        img.thumbnail((size, size))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=50)
        return f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode()}"
    except Exception as e:
        print(f"Error creating blur thumbnail: {e}")
        return None


def is_edit_request(message: str) -> bool:
    """Detect if user is requesting an edit to a previous image"""
    edit_keywords = [
        "add to", "edit", "modify", "change", "update", "remove from",
        "to this", "to it", "this image", "that image", "the image",
        "to the", "from the", "in the", "on the", "previous image",
        "first image", "second image", "third image", "last image",
        "image 1", "image 2", "image 3"
    ]
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in edit_keywords)


def is_feedback_or_complaint(message: str) -> tuple[bool, str]:
    """
    Detect if user is giving feedback/complaint about missing elements.
    Returns (is_feedback, interpreted_intent)

    "doesn't have cats" -> user WANTS cats, they're complaining it's missing
    "missing the dogs" -> user WANTS dogs
    """
    message_lower = message.lower()

    complaint_patterns = [
        (r"doesn'?t have (?:any )?(.+)", "missing"),
        (r"does not have (?:any )?(.+)", "missing"),
        (r"missing (?:the )?(.+)", "missing"),
        (r"where (?:are|is) (?:the )?(.+)", "missing"),
        (r"no (.+) in (?:the |this )?(?:video|image)", "missing"),
        (r"(.+) (?:is|are) missing", "missing"),
        (r"forgot (?:the |about )?(.+)", "missing"),
        (r"didn'?t include (?:the )?(.+)", "missing"),
        (r"left out (?:the )?(.+)", "missing"),
    ]

    for pattern, _ in complaint_patterns:
        match = re.search(pattern, message_lower)
        if match:
            missing_element = match.group(1).strip().rstrip('.')
            return True, missing_element

    return False, ""


async def enhance_prompt_with_cot(message: str, mode: str, aspect_ratio: str, image_context: list = None) -> tuple[str, str, list, int | None]:
    """
    Use Chain of Thought reasoning to analyze and enhance the user's prompt.
    Returns (enhanced_prompt, reasoning_summary, reasoning_steps, num_images)

    num_images is the explicit count if user specified one, None otherwise.

    If this is an edit request referencing previous images, skip enhancement
    and let the agent handle it with proper context.
    """
    global agent_app

    # Check if this is an edit request - don't enhance, let agent handle with context
    if is_edit_request(message):
        reasoning_steps = [
            {"type": "thought", "content": f"Detected edit request: \"{message[:60]}...\"" if len(message) > 60 else f"Detected edit request: \"{message}\""},
            {"type": "thought", "content": "Passing to agent with image context for editing"},
        ]
        return message, "Edit request - using original prompt with context", reasoning_steps, None

    # Build context about previous images if available
    image_context_str = ""
    if image_context:
        image_list = "\n".join([f"- Image {img['generation']}: \"{img['prompt'][:50]}...\"" for img in image_context])
        image_context_str = f"\n\nPrevious images in conversation:\n{image_list}\n\nIf user references these, don't create new subject - they want to modify existing."

    cot_prompt = f"""You are an expert prompt engineer for AI image generation. Analyze and enhance the user's prompt using step-by-step reasoning.

User's prompt: "{message}"
Mode: {mode} ({"high quality, detailed" if mode == "pro" else "fast generation"})
Aspect ratio: {aspect_ratio}{image_context_str}

Think through this step by step:

## Step 1: Core Subject Analysis
Identify the main subject(s) and their key characteristics.
IMPORTANT: Note if the user explicitly requested a specific number of images (e.g., "1 image", "5 images", "a single picture").

## Step 2: Missing Visual Elements
What important visual details are missing? Consider:
- Environment/setting (if not specified)
- Time of day and lighting conditions
- Artistic style or rendering approach
- Composition and framing for the aspect ratio

## Step 3: Enhancement Plan
List the specific additions that would improve the image quality without changing the user's intent.

## Step 4: Enhanced Prompt
Write the final enhanced prompt (under 80 words). Keep the user's original concept but add the visual details identified above.

Respond in this exact format:
REASONING: [Your step-by-step analysis in 2-3 sentences]
NUM_IMAGES: [Number if user explicitly specified one (1-4), otherwise "default"]
ENHANCED: [The final enhanced prompt]"""

    try:
        from smolagents.models import ChatMessage, MessageRole
        messages = [ChatMessage(role=MessageRole.USER, content=cot_prompt)]
        response = agent_app.model.generate(messages, max_tokens=400, temperature=0.7)
        content = response.content.strip()

        # Parse the structured response
        reasoning = ""
        enhanced = message  # fallback to original
        num_images = None

        if "REASONING:" in content and "ENHANCED:" in content:
            # Extract NUM_IMAGES if present
            if "NUM_IMAGES:" in content:
                num_images_part = content.split("NUM_IMAGES:")[1].split("ENHANCED:")[0].strip()
                # Parse the number if it's not "default"
                if num_images_part.lower() != "default":
                    try:
                        num_images = int(re.search(r'\d+', num_images_part).group())
                        # num_images = max(1, min(num_images, 4))  # Clamp to 1-4
                    except (ValueError, AttributeError):
                        num_images = None

            parts = content.split("ENHANCED:")
            reasoning_part = parts[0].replace("REASONING:", "").strip()
            # Remove NUM_IMAGES line from reasoning
            if "NUM_IMAGES:" in reasoning_part:
                reasoning_part = reasoning_part.split("NUM_IMAGES:")[0].strip()
            enhanced_part = parts[1].strip() if len(parts) > 1 else message

            reasoning = reasoning_part
            enhanced = enhanced_part

            # Clean up any markdown or extra formatting
            enhanced = enhanced.strip('"').strip("'").strip()
            if enhanced.startswith("**") and enhanced.endswith("**"):
                enhanced = enhanced[2:-2]

        elif "Enhanced prompt:" in content.lower():
            enhanced = content.split("Enhanced prompt:")[-1].strip()
            reasoning = "Analyzed prompt and added visual details"
        else:
            # Try to extract just the enhanced part if format wasn't followed
            enhanced = content.split("\n")[-1].strip() if "\n" in content else content
            reasoning = "Enhanced with visual details"

        reasoning_steps = [
            {"type": "thought", "content": f"Analyzing: \"{message[:50]}...\"" if len(message) > 50 else f"Analyzing: \"{message}\""},
            {"type": "thought", "content": reasoning},
        ]

        return enhanced, reasoning, reasoning_steps, num_images
    except Exception as e:
        print(f"Prompt enhancement failed: {e}")
        return message, None, [], None


async def generate_reflection_with_llm(message: str, image_paths: list, video_paths: list, reasoning_steps: list, total_time: float, mode: str) -> str:
    """
    Use the LLM to generate a conversational reflection on the generated result.
    The LLM receives the actual generated images/videos to analyze.
    """
    global agent_app
    
    media_paths = image_paths + video_paths
    if not media_paths:
        return f"Generation completed in {total_time:.1f}s. Feel free to ask for adjustments!"
    
    # Summarize reasoning steps
    reasoning_summary_parts = []
    for step in reasoning_steps[:5]:
        step_type = type(step).__name__
        if step_type == 'PlanningStep' and hasattr(step, 'plan'):
            reasoning_summary_parts.append("- Planned approach")
        elif step_type == 'ToolCall' and hasattr(step, 'name'):
            reasoning_summary_parts.append(f"- Used tool: {step.name}")
        elif step_type == 'ActionStep':
            if hasattr(step, 'observations') and step.observations:
                reasoning_summary_parts.append("- Executed action")
        elif step_type == 'FinalAnswerStep':
            reasoning_summary_parts.append("- Generated result")
    
    reasoning_summary = "\n".join(reasoning_summary_parts) if reasoning_summary_parts else "- Generated image"
    
    reflection_prompt = f"""You are an AI image/video generation assistant. You just completed a generation task.

Original request: "{message}"
Mode: {mode}
Time taken: {total_time:.1f} seconds

Key steps:
{reasoning_summary}

The generated image/video is attached for your review.

Provide a concise 1-2 sentence conversational reflection on the result. Comment on what worked well, the visual quality, or suggest what the user could try next. Be friendly and encouraging.

Example: "The warm golden tones really capture that magical hour feeling. The composition balances all elements harmoniously - feel free to ask for adjustments!"

Your reflection:"""
    
    try:
        from smolagents.models import ChatMessage, MessageRole

        # Build message with image attachment
        message_content = [{"type": "text", "text": reflection_prompt}]

        # Add the first image/video for LLM to analyze
        first_media = media_paths[0]
        if any(ext in first_media for ext in [".mp4", ".mov", ".webm"]):
            # For videos, we can't send directly to most LLMs, so describe it
            message_content.append({"type": "text", "text": f"\n[Video generated at: {first_media}]"})
        else:
            # For images, include the image for visual analysis
            message_content.append({"type": "image_url", "image_url": {"url": f"file://{first_media}"}})

        messages = [ChatMessage(role=MessageRole.USER, content=message_content)]

        # Use a simple reflection-specific system prompt (not the agent tools prompt)
        reflection_system = "You are a friendly AI assistant that provides brief, encouraging reflections on generated images and videos. Keep responses conversational and under 2 sentences."

        response = agent_app.model.generate(
            messages,
            max_tokens=200,
            temperature=0.7,
            system_prompt=reflection_system
        )
        return response.content.strip()
    except Exception as e:
        print(f"Error generating reflection: {e}")
        return f"Generation completed successfully in {total_time:.1f}s. The result looks great - feel free to ask for adjustments or try something new!"


async def generate_with_streaming(message: str, user_settings: dict, history: list = None) -> AsyncGenerator[str, None]:
    """
    Run the agent and stream REAL ReAct reasoning steps:
    - reasoning: Agent ACTUAL thinking process (from MultiStepAgent)
    - tool_call: When a tool is invoked (REAL tool calls)
    - image_preview: Blur thumbnail for progressive loading
    - image_complete: Image URL ready
    - video_complete: Video URL ready
    - reflection: Optional LLM reflection (async, non-blocking)
    - complete: Generation finished
    - error: If something goes wrong
    """
    global agent_app

    if not agent_app:
        yield SSEEvent.format("error", {"message": "Agent not initialized"})
        return

    # Build prompt with settings
    mode = user_settings.get("mode", "fast")
    aspect_ratio = user_settings.get("aspectRatio", "square")
    selected_image = user_settings.get("selectedImage")
    enable_reflection = True

    # Check if user is giving feedback about missing elements
    is_complaint, missing_element = is_feedback_or_complaint(message)
    feedback_context = ""
    if is_complaint:
        feedback_context = f"\n\n[USER FEEDBACK INTERPRETATION]\nThe user is expressing that '{missing_element}' is MISSING from the previous result. They WANT '{missing_element}' to be INCLUDED. This is a complaint, not a request to remove it. Please regenerate WITH '{missing_element}' included."

    # Extract PIL images from history WITH file paths and user prompts for agent context
    from PIL import Image
    context_images = []
    image_path_map = []  # List of {generation: int, path: str, prompt: str}
    generation_num = 1
    last_user_prompt = ""

    if history:
        for msg in history:
            if msg.get("role") == "user":
                last_user_prompt = msg.get("content", "")
            elif msg.get("role") == "assistant":
                msg_images = msg.get("images", [])
                if msg_images:
                    for img_url in msg_images:
                        if "path=" in img_url:
                            local_path = img_url.split("path=")[-1]
                            try:
                                img = Image.open(local_path)
                                context_images.append(img.copy())
                                image_path_map.append({
                                    "generation": generation_num,
                                    "path": local_path,
                                    "prompt": last_user_prompt
                                })
                                generation_num += 1
                            except Exception as e:
                                print(f"Could not load image {local_path}: {e}")

    has_previous_media = len(context_images) > 0

    # Build image reference context for agent (with actual paths!)
    settings_text = f"""[System Settings]
Performance Mode: {mode}
Aspect Ratio: {aspect_ratio}
Default Images: Generate 4 image variations by default (use num_images=4). If user explicitly requests a specific number, use that instead."""

    if has_previous_media:
        # Build clear image reference list with paths and original prompts
        image_refs = []
        for img_info in image_path_map:
            gen = img_info["generation"]
            path = img_info["path"]
            prompt = img_info["prompt"]
            # Include the original prompt so agent knows what each image contains
            prompt_preview = prompt[:80] + "..." if len(prompt) > 80 else prompt
            image_refs.append(f"  - Image {gen}: \"{prompt_preview}\" -> {path}")

        image_list = "\n".join(image_refs)

        # Check if user selected an image
        selected_path = None
        selected_gen = None
        if selected_image and selected_image.get("url"):
            url = selected_image["url"]
            if "path=" in url:
                selected_path = url.split("path=")[-1]
                # Find which generation this corresponds to
                for img_info in image_path_map:
                    if img_info["path"] == selected_path:
                        selected_gen = img_info["generation"]
                        break

        if selected_path:
            context_note = f"""[CONVERSATION IMAGE HISTORY]
Previously generated images in this conversation:
{image_list}

CURRENTLY SELECTED IMAGE: Image {selected_gen} ({selected_path})
The user has selected this image. When they say "edit this", "modify it", "change this image", "add to it", etc., use fal_image_edit with this path.

IMPORTANT: Match user references to the correct image:
- "the cat image" / "the one with the cat" -> find the image whose prompt mentions cats
- "the first/second/third image" -> Image 1/2/3
- "that image" / "it" / "this" -> the CURRENTLY SELECTED IMAGE above
- "the sunset" -> find the image whose prompt mentions sunset"""
        else:
            context_note = f"""[CONVERSATION IMAGE HISTORY]
Previously generated images in this conversation:
{image_list}

IMPORTANT: Match user references to the correct image based on content:
- "the cat image" / "the one with the cat" -> find the image whose prompt mentions cats
- "the first/second/third image" -> Image 1/2/3
- "that image" / "the previous image" -> the most recently generated image (highest number)
- When user wants to edit/modify an image, use fal_image_edit with the matched path."""

        current_message_text = f"{message}\n\n{settings_text}\n\n{context_note}{feedback_context}"
    else:
        current_message_text = f"{message}\n\n{settings_text}{feedback_context}"

    agent_error = None
    reasoning_steps = []

    def run_agent_with_streaming():
        """Run agent and capture streaming steps with multimodal support"""
        nonlocal agent_error, reasoning_steps
        try:
            full_prompt = f"[Context: Previous images are attached for your visual analysis]\n\n{current_message_text}" if context_images else current_message_text

            # Pass images directly to agent.run() for proper multimodal support
            for step in agent_app.agent.run(full_prompt, images=context_images if context_images else None, stream=True):
                reasoning_steps.append(step)
        except Exception as e:
            print(f"Error in agent execution: {e}")
            import traceback
            traceback.print_exc()
            agent_error = e

    # Enhance prompt with CoT reasoning (pass image context for awareness)
    yield SSEEvent.format("reasoning", {"content": "🧠 Analyzing your prompt..."})
    enhanced_prompt, reasoning_summary, cot_steps, explicit_num_images = await enhance_prompt_with_cot(
        message, mode, aspect_ratio, image_context=image_path_map if has_previous_media else None
    )

    # Stream the CoT reasoning steps
    for step in cot_steps:
        yield SSEEvent.format("reasoning", {"content": step["content"]})
        await asyncio.sleep(0.05)

    # Send enhanced prompt to frontend
    if enhanced_prompt != message:
        yield SSEEvent.format("enhanced_prompt", {"original": message, "enhanced": enhanced_prompt})
        yield SSEEvent.format("reasoning", {"content": f"✨ Final enhanced prompt ready"})

    # Update settings_text if user explicitly requested a specific number of images
    if explicit_num_images is not None:
        settings_text = f"""[System Settings]
Performance Mode: {mode}
Aspect Ratio: {aspect_ratio}
Number of Images: Generate exactly {explicit_num_images} image(s) as explicitly requested by the user (use num_images={explicit_num_images})."""

    # Update the prompt for the agent (preserve context_note with image paths)
    if has_previous_media:
        current_message_text = f"{enhanced_prompt}\n\n{settings_text}\n\n{context_note}{feedback_context}"
    else:
        current_message_text = f"{enhanced_prompt}\n\n{settings_text}{feedback_context}"

    yield SSEEvent.format("reasoning", {"content": "🤖 Starting generation..."})
    await asyncio.sleep(0.1)

    # Start agent in background thread with streaming
    start_time = time.time()
    agent_thread = threading.Thread(target=run_agent_with_streaming)
    agent_thread.start()

    last_step_count = 0
    image_paths = []
    video_paths = []

    async def process_step(step):
        """Process a single agent step and yield SSE events"""
        nonlocal image_paths, video_paths
        step_type = type(step).__name__

        if step_type == 'PlanningStep' and hasattr(step, 'plan'):
            plan_text = str(step.plan).strip()
            if plan_text:
                if "## 2. Plan" in plan_text:
                    plan_section = plan_text.split("## 2. Plan")[1].split("```")[0].strip()
                    yield SSEEvent.format("reasoning", {"content": f"📋 Planning:\n{plan_section[:400]}"})
                else:
                    yield SSEEvent.format("reasoning", {"content": "📋 Analyzing your request and planning..."})
                await asyncio.sleep(0.05)

        elif step_type == 'ToolCall' and hasattr(step, 'name'):
            tool_args = str(step.arguments) if hasattr(step, 'arguments') else ""
            friendly_tool = step.name
            if "fal_image_generation" in tool_args:
                friendly_tool = "Image Generation"
            elif "fal_video_generation" in tool_args:
                friendly_tool = "Video Generation"
            elif "fal_image_edit" in tool_args:
                friendly_tool = "Image Editing"

            yield SSEEvent.format("tool_call", {"tool": friendly_tool, "args": tool_args[:200]})
            yield SSEEvent.format("reasoning", {"content": f"🔧 Using {friendly_tool}..."})
            await asyncio.sleep(0.05)

        elif step_type == 'ActionStep':
            if hasattr(step, 'model_output') and step.model_output:
                reasoning_text = str(step.model_output).strip()
                if "Thought:" in reasoning_text:
                    thought = reasoning_text.split("Thought:")[1].split("<code>")[0].strip()
                    yield SSEEvent.format("reasoning", {"content": f"💭 {thought}"})
                    await asyncio.sleep(0.05)

            if hasattr(step, 'observations') and step.observations:
                observation = str(step.observations)
                new_images = parse_image_paths(observation)
                for img_path in new_images:
                    if img_path not in image_paths:
                        image_paths.append(img_path)
                        yield SSEEvent.format("reasoning", {"content": "✅ Image generated successfully"})
                        thumbnail = create_blur_thumbnail(img_path)
                        if thumbnail:
                            yield SSEEvent.format("image_preview", {"blur_data": thumbnail})
                            await asyncio.sleep(0.05)
                        yield SSEEvent.format("image_complete", {"url": f"/api/media?path={img_path}"})
                        await asyncio.sleep(0.05)

                new_videos = parse_video_paths(observation)
                for vid_path in new_videos:
                    if vid_path not in video_paths:
                        video_paths.append(vid_path)
                        yield SSEEvent.format("reasoning", {"content": "✅ Video generated successfully"})
                        yield SSEEvent.format("video_complete", {"url": f"/api/media?path={vid_path}"})
                        await asyncio.sleep(0.05)

        elif step_type == 'FinalAnswerStep':
            if hasattr(step, 'output'):
                output_str = str(step.output)
                # Extract any images from final answer
                final_images = parse_image_paths(output_str)
                for img_path in final_images:
                    if img_path not in image_paths:
                        image_paths.append(img_path)
                        yield SSEEvent.format("image_complete", {"url": f"/api/media?path={img_path}"})

                # Send text response to UI (agent's message to user)
                # Remove image paths from text to get clean message
                clean_output = output_str
                for img_path in final_images:
                    clean_output = clean_output.replace(img_path, "").strip()
                clean_output = clean_output.replace("Generated image(s):", "").strip()
                clean_output = clean_output.replace(",", "").strip()

                if clean_output and len(clean_output) > 5:
                    yield SSEEvent.format("agent_message", {"content": clean_output})

            yield SSEEvent.format("reasoning", {"content": "✅ Complete!"})
            await asyncio.sleep(0.05)

    # Stream reasoning as it happens
    while agent_thread.is_alive():
        if len(reasoning_steps) > last_step_count:
            for step in reasoning_steps[last_step_count:]:
                async for event in process_step(step):
                    yield event
            last_step_count = len(reasoning_steps)
        await asyncio.sleep(0.1)

    agent_thread.join()

    # Process any remaining steps after thread finishes
    if len(reasoning_steps) > last_step_count:
        for step in reasoning_steps[last_step_count:]:
            async for event in process_step(step):
                yield event

    total_time = time.time() - start_time

    # Handle errors
    if agent_error:
        yield SSEEvent.format("error", {"message": str(agent_error)})
        return

    # Complete event (don't block on reflection)
    yield SSEEvent.format("complete", {"duration": total_time})

    # Optional: Generate reflection AFTER completion (non-blocking)
    if enable_reflection and (image_paths or video_paths):
        try:
            reflection = await generate_reflection_with_llm(
                message=message,
                image_paths=image_paths,
                video_paths=video_paths,
                reasoning_steps=reasoning_steps,
                total_time=total_time,
                mode=mode
            )
            yield SSEEvent.format("reflection", {"content": reflection})
        except Exception as e:
            print(f"Error generating reflection (non-critical): {e}")


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Stream chat responses via Server-Sent Events"""
    return StreamingResponse(
        generate_with_streaming(request.message, request.settings, request.history),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/api/media")
async def serve_media(path: str):
    """Serve generated media files (with security validation)"""
    try:
        # Security: Resolve the path and check it's in temp directory
        resolved_path = Path(path).resolve()
        temp_dir = Path(tempfile.gettempdir()).resolve()

        # Ensure the path is within temp directory (prevent path traversal)
        if not str(resolved_path).startswith(str(temp_dir)):
            raise HTTPException(status_code=403, detail="Access denied")

        if not resolved_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        # Validate file extension
        ext = resolved_path.suffix.lower()
        content_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".mp4": "video/mp4",
            ".webm": "video/webm"
        }

        if ext not in content_types:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        return FileResponse(
            str(resolved_path),
            media_type=content_types[ext]
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error serving media: {e}")
        raise HTTPException(status_code=500, detail="Error serving file")


@app.get("/api/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok", "agent_ready": agent_app is not None}


# Serve frontend index.html for all non-API routes (production only)
if PRODUCTION_MODE:
    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        """Serve the React app for all non-API routes"""
        frontend_dist = Path(__file__).parent / "dist"
        index_file = frontend_dist / "index.html"

        # If path starts with api, return 404
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API endpoint not found")

        # Check if file exists in dist
        requested_file = frontend_dist / full_path
        if requested_file.exists() and requested_file.is_file():
            return FileResponse(requested_file)

        # Otherwise serve index.html (for client-side routing)
        if index_file.exists():
            return FileResponse(index_file)

        raise HTTPException(status_code=404, detail="Frontend not built")


if __name__ == "__main__":
    import uvicorn
    print("🌟 Starting Magic Hour AI Server...")
    print("📡 API: http://localhost:8000")
    print("🎨 Frontend: http://localhost:5173")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
