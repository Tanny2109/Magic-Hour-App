"""Utility functions for streaming agent responses to Gradio"""
from __future__ import annotations

import os
import re
from typing import Generator

from gradio import ChatMessage
from smolagents import CodeAgent


def parse_image_paths(text: str) -> list[str]:
    """Extract valid image paths or URLs from agent output
    
    Args:
        text: Agent output text potentially containing image paths
        
    Returns:
        List of valid image file paths or URLs
    """
    # Support multiple common image formats, both local paths and URLs
    pattern = r'((?:/|https?://)[^\s]+\.(?:png|jpg|jpeg|webp))'
    matches = re.findall(pattern, text)
    # Remove duplicates while preserving order
    unique_matches = list(dict.fromkeys(matches))
    
    valid_paths = []
    for path in unique_matches:
        if path.startswith(("http://", "https://")):
            valid_paths.append(path)
        elif os.path.exists(path):
            valid_paths.append(path)
            
    return valid_paths


def parse_video_paths(text: str) -> list[str]:
    """Extract valid video paths or URLs from agent output
    
    Args:
        text: Agent output text potentially containing video paths
        
    Returns:
        List of valid video file paths or URLs
    """
    # Support common video formats, both local paths and URLs
    pattern = r'((?:/|https?://)[^\s]+\.(?:mp4|mov|avi|webm))'
    matches = re.findall(pattern, text)
    # Remove duplicates while preserving order
    unique_matches = list(dict.fromkeys(matches))
    
    valid_paths = []
    for path in unique_matches:
        if path.startswith(("http://", "https://")):
            valid_paths.append(path)
        elif os.path.exists(path):
            valid_paths.append(path)
            
    return valid_paths


def pull_message_from_step(step_log: dict) -> Generator[ChatMessage, None, None]:
    """Extract ChatMessages from a single agent step log
    
    Args:
        step_log: Dictionary containing agent step information
        
    Yields:
        ChatMessage objects for each component of the step
    """
    # Display task
    if step_log.get("task"):
        yield ChatMessage(
            role="assistant",
            metadata={"title": "📋 Task"},
            content=step_log["task"]
        )

    # Display rationale/thinking
    if step_log.get("rationale"):
        yield ChatMessage(
            role="assistant",
            metadata={"title": "💭 Thinking"},
            content=step_log["rationale"]
        )

    # Display LLM output
    if step_log.get("llm_output"):
        yield ChatMessage(
            role="assistant",
            content=step_log["llm_output"]
        )

    # Display tool calls
    if step_log.get("tool_calls"):
        for tool_call in step_log["tool_calls"]:
            tool_name = tool_call.get("name", "unknown")
            tool_args = tool_call.get("arguments", {})

            content = f"```json\n{tool_args}\n```"
            yield ChatMessage(
                role="assistant",
                metadata={"title": f"🛠️ Using tool: {tool_name}"},
                content=content
            )

    # Display observations/results
    if step_log.get("observation"):
        yield ChatMessage(
            role="assistant",
            metadata={"title": "👁️ Observation"},
            content=f"```\n{step_log['observation']}\n```"
        )

    # Display errors
    if step_log.get("error"):
        yield ChatMessage(
            role="assistant",
            metadata={"title": "💥 Error"},
            content=str(step_log["error"])
        )


def stream_from_smolagent(
    agent: CodeAgent, prompt: str
) -> Generator[ChatMessage, None, ChatMessage | None]:
    """Run a smolagent and stream messages as ChatMessages
    
    Args:
        agent: The CodeAgent instance to run
        prompt: User's input prompt
        
    Yields:
        ChatMessage objects for each step of the agent's execution
        
    Returns:
        Final ChatMessage with the agent's output (text and/or images)
    """
    # Clear previous logs
    agent.logs = []

    # Initial thinking message
    yield ChatMessage(
        role="assistant",
        content="🤔 Analyzing your request..."
    )

    # Run agent - it populates agent.logs
    agent_output = agent.run(prompt)
    print(f"Agent output: {agent_output}")

    # Stream thinking from logs
    for step_log in agent.logs:
        if isinstance(step_log, dict):
            for message in pull_message_from_step(step_log):
                yield message

    # Parse output for media
    image_paths = parse_image_paths(str(agent_output))
    video_paths = parse_video_paths(str(agent_output))
    
    # Get the text content without the file paths (roughly)
    output_text = str(agent_output).strip()
    
    # Yield text response if it's substantial or if there are no media files
    if output_text and (not (image_paths or video_paths) or len(output_text) > 200):
         yield ChatMessage(
            role="assistant",
            metadata={"title": "✅ Final Answer"},
            content=output_text
        )

    # Yield images
    for img_path in image_paths:
        yield ChatMessage(
            role="assistant",
            content={"path": img_path, "mime_type": "image/png"}
        )
        
    # Yield videos
    for video_path in video_paths:
        yield ChatMessage(
            role="assistant",
            content={"path": video_path, "mime_type": "video/mp4"}
        )
        
    # Fallback if we have media but didn't yield text above, and the text is short/just the path
    # We might want to show the text if it contains other info, but usually it's just "Generated image at ..."
    # So we skip it if we showed media, unless it was long.
    
    if not (image_paths or video_paths) and not output_text:
         yield ChatMessage(
            role="assistant",
            metadata={"title": "✅ Done"},
            content="Task completed."
        )

    return None
