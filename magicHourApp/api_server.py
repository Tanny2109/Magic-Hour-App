"""
FastAPI backend for Magic Hour AI - streams agent reasoning and results via SSE
LangGraph version: Uses LangChain/LangGraph with fal.ai LLM
"""
import fal_client
from dotenv import load_dotenv
import os
import sys
import json
import asyncio
import threading
import time
import tempfile
import io
import re
import base64
from pathlib import Path
from typing import AsyncGenerator, Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image

# Add the mh_langgraph_workflow to path
WORKFLOW_PATH = Path(__file__).parent.parent / "mh_langgraph_workflow"
sys.path.insert(0, str(WORKFLOW_PATH))

# Load environment variables
load_dotenv(WORKFLOW_PATH / ".env")

# Configure fal.ai
fal_client.api_key = os.getenv("FAL_KEY")

# Import from mh_langgraph_workflow
from src.agents import create_agent, ContentAgent
from src.models import FalAILLM

# Global agent instance
agent: Optional[ContentAgent] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize agent on startup"""
    global agent
    print("🚀 Initializing Magic Hour AI Agent (LangGraph)...")
    try:
        agent = create_agent(
            fal_model_name=os.getenv("FAL_MODEL_NAME", "google/gemini-2.5-flash"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096")),
        )
        print(f"✅ Agent initialized with model: {os.getenv('FAL_MODEL_NAME', 'google/gemini-2.5-flash')}")
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        raise
    yield
    print("👋 Shutting down...")


app = FastAPI(title="Magic Hour AI (LangGraph)", lifespan=lifespan)

# Detect production mode
PRODUCTION_MODE = os.getenv("PRODUCTION", "false").lower() == "true"

# CORS configuration
allowed_origins = ["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"]
if PRODUCTION_MODE:
    production_domain = os.getenv("FRONTEND_URL", "*")
    allowed_origins = ["*"] if production_domain == "*" else allowed_origins + [production_domain]

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
    history: list = []
    session_id: Optional[str] = None  # Optional: Frontend provides to maintain conversation history


class SSEEvent:
    @staticmethod
    def format(event_type: str, data: dict) -> str:
        return f"data: {json.dumps({'type': event_type, **data})}\n\n"


def create_blur_thumbnail(image_path: str, size: int = 20) -> Optional[str]:
    """Create tiny base64 thumbnail for blur effect"""
    try:
        img = Image.open(image_path)
        img.thumbnail((size, size))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=50)
        return f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode()}"
    except Exception as e:
        print(f"Error creating blur thumbnail: {e}")
        return None


def extract_images_from_history(history: list) -> tuple[List[Image.Image], List[dict], List[str]]:
    """
    Extract PIL images, metadata, and paths from conversation history.
    Returns: (pil_images, image_metadata, image_paths)
    """
    pil_images = []
    image_metadata = []
    image_paths = []
    last_user_prompt = ""

    if not history:
        return [], [], []

    for msg in history:
        if msg.get("role") == "user":
            last_user_prompt = msg.get("content", "")
        elif msg.get("role") == "assistant":
            for img_url in msg.get("images", []):
                if "path=" in img_url:
                    local_path = img_url.split("path=")[-1]
                    try:
                        img = Image.open(local_path)
                        pil_images.append(img.copy())
                        image_paths.append(local_path)
                        image_metadata.append({
                            "index": len(pil_images),
                            "path": local_path,
                            "prompt": last_user_prompt
                        })
                    except Exception as e:
                        print(f"Could not load image {local_path}: {e}")

    return pil_images, image_metadata, image_paths


def parse_image_paths(text: str) -> List[str]:
    """Extract image file paths from text."""
    patterns = [
        r'(/tmp/[^\s\'"]+\.(?:png|jpg|jpeg|webp))',
        r'(/var/folders/[^\s\'"]+\.(?:png|jpg|jpeg|webp))',
    ]
    paths = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        paths.extend(matches)
    return [p for p in set(paths) if os.path.exists(p)]


def parse_video_paths(text: str) -> List[str]:
    """Extract video file paths from text."""
    patterns = [
        r'(/tmp/[^\s\'"]+\.(?:mp4|webm|mov))',
        r'(/var/folders/[^\s\'"]+\.(?:mp4|webm|mov))',
    ]
    paths = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        paths.extend(matches)
    return [p for p in set(paths) if os.path.exists(p)]


async def generate_with_streaming(message: str, user_settings: dict, history: list = None, session_id: str = None) -> AsyncGenerator[str, None]:
    """
    Run the LangGraph agent and stream results via SSE.
    """
    global agent

    if not agent:
        yield SSEEvent.format("error", {"message": "Agent not initialized"})
        return

    # Use session_id if provided, otherwise create one from timestamp
    # IMPORTANT: Frontend should maintain session_id to preserve conversation history
    if not session_id:
        session_id = f"session-{int(time.time() * 1000)}"
        print(f"DEBUG - Created new session ID: {session_id}")
    else:
        print(f"DEBUG - Using existing session ID: {session_id}")

    mode = user_settings.get("mode", "fast")
    aspect_ratio = user_settings.get("aspectRatio", "square")
    selected_image = user_settings.get("selectedImage")

    # Extract images from history
    pil_images, image_metadata, image_paths = extract_images_from_history(history)
    has_context = len(pil_images) > 0

    # Settings block
    settings_block = f"""[Settings]
Mode: {mode} | Aspect Ratio: {aspect_ratio}
Default: Generate 4 variations unless user specifies otherwise."""

    # Build image paths reference for agent context
    image_paths_ref = ""
    if image_metadata:
        paths_list = "\n".join([f"  - Image {m['index']}: {m['path']}" for m in image_metadata])
        image_paths_ref = f"\n\n[Image File Paths]\n{paths_list}"

    # Just send the user message + settings
    # The agent will handle visual analysis internally if needed
    yield SSEEvent.format("reasoning", {"content": "🧠 Processing your request..."})

    full_prompt = f"""{message}"""

    yield SSEEvent.format("reasoning", {"content": "🤖 Starting generation..."})

    start_time = time.time()
    agent_result = None
    agent_error = None
    generated_images = []
    generated_videos = []

    def run_agent():
        nonlocal agent_result, agent_error
        try:
            # Use the session_id for thread continuity
            agent_result = agent.invoke(
                message=full_prompt,
                image_paths=image_paths if image_paths else None,
                thread_id=session_id,
                settings=user_settings,
            )
        except Exception as e:
            print(f"Error in agent execution: {e}")
            import traceback
            traceback.print_exc()
            agent_error = e

    # Run agent in background thread
    agent_thread = threading.Thread(target=run_agent)
    agent_thread.start()

    # Poll for completion
    while agent_thread.is_alive():
        await asyncio.sleep(0.2)

    agent_thread.join()

    if agent_error:
        yield SSEEvent.format("error", {"message": str(agent_error)})
        return

    # Process the result
    if agent_result:
        all_messages = agent_result.get("messages", [])
        generation_history = agent_result.get("generation_history", [])
        
        # Only process NEW messages added in this turn
        # We look for the last HumanMessage and take everything after it
        new_messages = []
        for msg in reversed(all_messages):
            if type(msg).__name__ == "HumanMessage":
                break
            new_messages.insert(0, msg)

        # Extract final response and tool results
        final_response = ""
        reasoning_steps = []  # Collect all reasoning steps
        visual_analysis_content = None

        for msg in new_messages:
            msg_content = msg.content if hasattr(msg, 'content') else str(msg)
            msg_type = type(msg).__name__

            # Handle custom message types
            if msg_type == "ReasoningMessage":
                # Reasoning/thinking message - add to collapsible dropdown
                reasoning_steps.append(str(msg_content))
                yield SSEEvent.format("reasoning_step", {
                    "content": str(msg_content),
                    "collapsible": True
                })
                await asyncio.sleep(0.01)

            elif msg_type == "VisualAnalysisMessage":
                # Visual context analysis
                visual_analysis_content = str(msg_content)
                yield SSEEvent.format("visual_analysis", {
                    "content": str(msg_content),
                    "collapsible": True
                })
                await asyncio.sleep(0.01)

            elif msg_type == "AIMessage":
                content_str = str(msg_content)

                # Only extract paths from TOOL RESULT messages (not tool calls)
                # Tool results contain "Successfully" while tool calls contain the original paths
                is_tool_result = "Successfully" in content_str

                if is_tool_result:
                    found_images = parse_image_paths(content_str)
                    found_videos = parse_video_paths(content_str)

                    for img_path in found_images:
                        if img_path not in generated_images:
                            generated_images.append(img_path)
                            yield SSEEvent.format("image_complete", {"url": f"/api/media?path={img_path}"})
                            await asyncio.sleep(0.05)

                    for vid_path in found_videos:
                        if vid_path not in generated_videos:
                            generated_videos.append(vid_path)
                            yield SSEEvent.format("video_complete", {"url": f"/api/media?path={vid_path}"})
                            await asyncio.sleep(0.05)

                # Check for tool calls in response (agent reasoning)
                elif '{"tool":' in content_str or '"tool"' in content_str:
                    # Extract the reasoning part before the tool call
                    if "[REASONING]" in content_str:
                        # Extract reasoning between [REASONING] and the tool call
                        import re
                        reasoning_match = re.search(r'\[REASONING\](.*?)(?=\[ACTION\]|\{|```)', content_str, re.DOTALL)
                        if reasoning_match:
                            reasoning = reasoning_match.group(1).strip()
                            if reasoning and reasoning not in reasoning_steps:
                                reasoning_steps.append(reasoning)
                                yield SSEEvent.format("reasoning_step", {
                                    "content": reasoning,
                                    "collapsible": True
                                })
                    elif "```json" in content_str:
                        reasoning = content_str.split("```json")[0].strip()
                        if reasoning and reasoning not in reasoning_steps:
                            reasoning_steps.append(reasoning)
                            yield SSEEvent.format("reasoning_step", {
                                "content": reasoning[:500],
                                "collapsible": True
                            })
                else:
                    # This is a conversational response (no tool call)
                    final_response = content_str

        # If no images were found in new AIMessages, check the LATEST batch from generation_history only
        if not generated_images and not generated_videos and generation_history:
            latest_batch = generation_history[-1]
            if latest_batch["type"] == "image":
                for path in latest_batch["paths"]:
                    if os.path.exists(path) and path not in generated_images:
                        generated_images.append(path)
                        thumbnail = create_blur_thumbnail(path)
                        if thumbnail:
                            yield SSEEvent.format("image_preview", {"blur_data": thumbnail})
                        yield SSEEvent.format("image_complete", {"url": f"/api/media?path={path}"})
            elif latest_batch["type"] == "video":
                for path in latest_batch["paths"]:
                    if os.path.exists(path) and path not in generated_videos:
                        generated_videos.append(path)
                        yield SSEEvent.format("video_complete", {"url": f"/api/media?path={path}"})

        # Send final message if present
        if final_response:
            # Clean up the response
            clean_response = final_response
            for path in generated_images + generated_videos:
                clean_response = clean_response.replace(path, "").strip()

            # Remove tool call JSON if present
            if "```json" in clean_response:
                clean_response = clean_response.split("```json")[0].strip()

            if clean_response and len(clean_response) > 5:
                yield SSEEvent.format("agent_message", {"content": clean_response})

    total_time = time.time() - start_time
    yield SSEEvent.format("reasoning", {"content": "✅ Complete!"})
    yield SSEEvent.format("complete", {
        "duration": total_time,
        "session_id": session_id  # Send session_id back to frontend
    })


@app.post("/api/chat")
async def chat(request: ChatRequest):
    return StreamingResponse(
        generate_with_streaming(
            request.message,
            request.settings,
            request.history,
            request.session_id
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/api/media")
async def serve_media(path: str):
    """Serve generated media files with security validation"""
    try:
        resolved_path = Path(path).resolve()
        temp_dir = Path(tempfile.gettempdir()).resolve()

        if not str(resolved_path).startswith(str(temp_dir)):
            raise HTTPException(status_code=403, detail="Access denied")

        if not resolved_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        content_types = {
            ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".webp": "image/webp", ".mp4": "video/mp4", ".webm": "video/webm"
        }
        ext = resolved_path.suffix.lower()
        if ext not in content_types:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        return FileResponse(str(resolved_path), media_type=content_types[ext])
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error serving media: {e}")
        raise HTTPException(status_code=500, detail="Error serving file")


@app.get("/api/health")
async def health():
    return {"status": "ok", "agent_ready": agent is not None, "backend": "langgraph"}


if PRODUCTION_MODE:
    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        frontend_dist = Path(__file__).parent / "dist"
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API endpoint not found")

        requested_file = frontend_dist / full_path
        if requested_file.exists() and requested_file.is_file():
            return FileResponse(requested_file)

        index_file = frontend_dist / "index.html"
        if index_file.exists():
            return FileResponse(index_file)

        raise HTTPException(status_code=404, detail="Frontend not built")


if __name__ == "__main__":
    import uvicorn
    print("Starting Magic Hour AI Server (LangGraph)...")
    print("API: http://localhost:8000")
    print("Frontend: http://localhost:5173")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
