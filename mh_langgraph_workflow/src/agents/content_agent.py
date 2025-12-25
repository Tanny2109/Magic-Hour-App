"""LangGraph-based content generation agent with multimodal support using fal.ai LLM."""
import os
import base64
import re
import json
import time
from typing import Annotated, TypedDict, Sequence, Optional, List
from dataclasses import dataclass, field

import fal_client

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from ..models import FalAILLM
from ..tools import generate_images, edit_images, generate_video


# System prompt for the content generation agent
SYSTEM_PROMPT = """You are Magic Hour, an AI image/video generation assistant.

## Workflow: Think → Act

For EVERY request, follow this pattern:

1. **[REASONING]** - Think step by step:
   - What is the user asking for?
   - If [VISUAL CONTEXT ANALYSIS] is provided, what does it tell me?
   - Should I GENERATE new or EDIT existing?
   - Which images (if editing)? What prompt to use?

2. **[ACTION]** - Execute the tool call

## Visual Context Integration

When you receive [VISUAL CONTEXT ANALYSIS], this is a vision model's interpretation of recent images. USE IT to:
- Understand themes/characters (e.g., "Mortal Kombat characters detected")
- Interpret ambiguous requests (e.g., "scorpio" after "Sub-Zero" = Scorpion character)
- Decide whether to EDIT (maintain theme) or GENERATE NEW
- Choose which image paths to edit

## Tools

1. **generate_images** - Create NEW images
   - Use when: User wants something new OR theme shift from context
   - Args: prompt (detailed), mode ("fast"/"pro"), num_images (1-4)

2. **edit_images** - Modify EXISTING images
   - Use when: User says "add/edit/change" AND visual context exists
   - Args: image_paths (select from [Image Paths]), prompt (what to change)

3. **generate_video** - Create videos
   - Args: prompt (detailed description)

## Response Format

First output your reasoning:
```
[REASONING]
The user wants to add scorpio. From [VISUAL CONTEXT ANALYSIS], I can see the previous images show Sub-Zero from Mortal Kombat. Therefore "scorpio" likely refers to Scorpion, another MK character. I should EDIT the most recent Sub-Zero images to add Scorpion in the same style.
```

Then output the tool call:
```json
{"tool": "edit_images", "args": {"image_paths": [...paths from context...], "prompt": "add Scorpion from Mortal Kombat"}}
```

## Key Rules
- ALWAYS show [REASONING] first, tool call second
- Use [VISUAL CONTEXT ANALYSIS] to interpret ambiguous requests
- Edit = recent images only (unless user specifies "all" or specific batch)
- Generate = fresh content"""


class GenerationRecord(TypedDict):
    """Record of a single generation batch."""
    prompt: str
    paths: List[str]
    type: str  # "image" or "video"
    description: str  # Visual description of what was generated (cached)


class ReasoningMessage(BaseMessage):
    """Custom message type for agent reasoning/thinking."""
    type: str = "reasoning"

    def __init__(self, content: str, **kwargs):
        super().__init__(content=content, **kwargs)


class VisualAnalysisMessage(BaseMessage):
    """Custom message type for visual context analysis."""
    type: str = "visual_analysis"

    def __init__(self, content: str, **kwargs):
        super().__init__(content=content, **kwargs)


class AgentState(TypedDict):
    """State for the content generation agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    generated_content: List[str]  # All paths (for backward compat)
    generation_history: List[GenerationRecord]  # Structured history with prompts
    settings: dict
    pending_tool_call: Optional[dict]


def _image_to_base64(image_path: str) -> Optional[str]:
    """Convert a local image file to base64 data URL."""
    if not os.path.exists(image_path):
        return None

    with open(image_path, "rb") as f:
        image_data = f.read()

    if image_path.lower().endswith(".png"):
        mime_type = "image/png"
    elif image_path.lower().endswith((".jpg", ".jpeg")):
        mime_type = "image/jpeg"
    elif image_path.lower().endswith(".webp"):
        mime_type = "image/webp"
    else:
        mime_type = "image/png"

    base64_data = base64.b64encode(image_data).decode("utf-8")
    return f"data:{mime_type};base64,{base64_data}"


def _extract_image_paths(text: str) -> List[str]:
    """Extract image file paths from text."""
    patterns = [
        r'(/tmp/[^\s\'"]+\.(?:png|jpg|jpeg|webp))',
        r'(/var/folders/[^\s\'"]+\.(?:png|jpg|jpeg|webp))',
        r'(/[^\s\'"]+\.(?:png|jpg|jpeg|webp))',
    ]

    paths = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        paths.extend(matches)

    return [p for p in set(paths) if os.path.exists(p)]


def _extract_video_paths(text: str) -> List[str]:
    """Extract video file paths from text."""
    patterns = [
        r'(/tmp/[^\s\'"]+\.(?:mp4|webm|mov))',
        r'(/var/folders/[^\s\'"]+\.(?:mp4|webm|mov))',
        r'(/[^\s\'"]+\.(?:mp4|webm|mov))',
    ]

    paths = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        paths.extend(matches)

    return [p for p in set(paths) if os.path.exists(p)]


def _needs_visual_context(user_message: str) -> bool:
    """Detect if user message requires visual context analysis.

    Uses semantic indicators rather than hardcoded keywords.
    """
    message_lower = user_message.lower()

    # Indicators that suggest referring to previous content
    referential_patterns = [
        # Edit/modify actions
        "add ", "edit", "change", "modify", "update", "fix",
        "remove", "delete", "replace", "adjust",
        # Demonstratives/references
        "this", "that", "these", "those", "the ",
        "same", "similar", "like the",
        # Continuation indicators
        "more", "another", "also", "too",
        # Ambiguous terms that need context
        " it ", "them",
    ]

    return any(pattern in message_lower for pattern in referential_patterns)


def _find_relevant_images(
    user_message: str,
    generation_history: List[GenerationRecord],
    max_images: int = 4
) -> tuple[List[str], List[dict]]:
    """Find relevant images based on user message.

    Smart lookup that:
    1. Checks for explicit batch references ("batch 1", "the first one")
    2. Searches prompts for keyword matches ("the dragon" -> finds dragon batch)
    3. Falls back to latest batch if no specific reference found

    Returns:
        tuple: (image_paths, batch_metadata)
    """
    if not generation_history:
        return [], []

    message_lower = user_message.lower()
    matched_batches = []

    # 1. Check for explicit batch/position references
    batch_patterns = [
        (r'batch\s*(\d+)', lambda m: int(m.group(1)) - 1),  # "batch 1" -> index 0
        (r'(\d+)(?:st|nd|rd|th)\s*(?:image|one|batch)', lambda m: int(m.group(1)) - 1),  # "1st image"
        (r'(?:the\s+)?first\s+(?:image|one|batch)', lambda m: 0),
        (r'(?:the\s+)?second\s+(?:image|one|batch)', lambda m: 1),
        (r'(?:the\s+)?third\s+(?:image|one|batch)', lambda m: 2),
        (r'(?:the\s+)?last\s+(?:image|one|batch)', lambda m: -1),
        (r'(?:the\s+)?previous\s+(?:image|one|batch)', lambda m: -2),
        (r'(?:the\s+)?earlier\s+(?:image|one|batch)', lambda m: 0),  # Assume first
    ]

    for pattern, idx_func in batch_patterns:
        match = re.search(pattern, message_lower)
        if match:
            try:
                idx = idx_func(match)
                if idx < 0:
                    idx = len(generation_history) + idx
                if 0 <= idx < len(generation_history):
                    matched_batches.append(idx)
                    print(f"DEBUG - Explicit batch reference found: batch {idx + 1}")
            except (ValueError, IndexError):
                pass

    # 2. Search prompts AND descriptions for keyword matches (only if no explicit reference)
    if not matched_batches:
        # Extract potential keywords from user message (nouns, descriptors)
        # Simple approach: look for words > 3 chars that aren't common words
        stop_words = {
            "the", "add", "edit", "change", "make", "with", "this", "that",
            "these", "those", "from", "into", "more", "less", "like", "same",
            "image", "images", "picture", "pictures", "photo", "photos",
            "please", "could", "would", "should", "want", "need", "can",
        }

        words = re.findall(r'\b[a-z]{4,}\b', message_lower)
        keywords = [w for w in words if w not in stop_words]

        if keywords:
            print(f"DEBUG - Searching prompts/descriptions for keywords: {keywords}")

            for idx, record in enumerate(generation_history):
                prompt_lower = record["prompt"].lower()
                description_lower = record.get("description", "").lower()
                searchable_text = f"{prompt_lower} {description_lower}"

                # Check if any keyword matches the prompt or description
                for keyword in keywords:
                    if keyword in searchable_text:
                        if idx not in matched_batches:
                            matched_batches.append(idx)
                            match_source = "description" if keyword in description_lower else "prompt"
                            print(f"DEBUG - Keyword '{keyword}' matched batch {idx + 1} ({match_source})")
                        break

    # 3. Fall back to latest batch if nothing matched
    if not matched_batches:
        matched_batches = [len(generation_history) - 1]
        print(f"DEBUG - No specific reference found, using latest batch")

    # Collect images from matched batches
    image_paths = []
    batch_metadata = []

    for batch_idx in matched_batches:
        record = generation_history[batch_idx]
        if record["type"] == "image":
            for path in record["paths"]:
                if os.path.exists(path) and len(image_paths) < max_images:
                    image_paths.append(path)
                    batch_metadata.append({
                        "batch": batch_idx + 1,
                        "prompt": record["prompt"],
                        "description": record.get("description", ""),
                        "path": path
                    })

    print(f"DEBUG - Found {len(image_paths)} relevant images from {len(matched_batches)} batch(es)")
    return image_paths, batch_metadata


def _format_visual_context(
    batch_metadata: List[dict],
    user_message: str
) -> str:
    """Format cached image descriptions as context for the agent.

    Uses pre-computed descriptions from generation time - no vision API call needed.

    Args:
        batch_metadata: List of dicts with batch/prompt/description info for each image
        user_message: The user's current request

    Returns:
        Formatted context text for the agent.
    """
    if not batch_metadata:
        return ""

    print(f"DEBUG - Formatting visual context from {len(batch_metadata)} cached descriptions")

    # Build context from cached descriptions
    context_lines = []
    for i, meta in enumerate(batch_metadata):
        batch_num = meta['batch']
        description = meta.get('description', '')
        path = meta.get('path', '')

        if description:
            context_lines.append(f"**Batch {batch_num}** ({path}):\n  {description}")
        else:
            # Fallback to prompt if no description
            context_lines.append(f"**Batch {batch_num}** ({path}):\n  Generated from: {meta['prompt'][:100]}")

    context = "\n\n".join(context_lines)

    analysis = f"""## Images in Context

{context}

## User Request
"{user_message}"

## Available Paths for Editing
{chr(10).join(f"- {meta['path']}" for meta in batch_metadata)}"""

    print(f"DEBUG - Visual context formatted (no API call needed)")
    return analysis


def _describe_generated_images(
    image_paths: List[str],
    prompt_used: str,
    model_name: str = "google/gemini-2.5-flash"
) -> str:
    """Describe generated images right after creation.

    This runs once per generation and caches the description for future reference.
    Fast and concise - just captures what's visually present.
    """
    if not image_paths:
        return ""

    # Only analyze first image (they're usually variations of same prompt)
    image_path = image_paths[0]
    if not os.path.exists(image_path):
        return f"Generated from prompt: {prompt_used}"

    print(f"DEBUG - Describing generated image: {image_path}")

    try:
        # Upload image
        with open(image_path, "rb") as f:
            image_data = f.read()
        content_type = "image/png" if image_path.lower().endswith(".png") else "image/jpeg"
        image_url = fal_client.upload(image_data, content_type)

        # Get concise description
        st = time.time()
        result = fal_client.subscribe(
            "openrouter/router/vision",
            arguments={
                "model": model_name,
                "image_url": [image_url],
                "prompt": f"""Describe this AI-generated image in 1-2 sentences.

Original prompt was: "{prompt_used}"

Focus on:
- Main subject (character, object, scene)
- Notable visual style or aesthetic
- Key details that would help identify this image later

Be specific but brief. Example: "A fierce dragon with orange scales breathing blue fire, fantasy digital art style with dramatic lighting."
""",
                "system_prompt": "You describe images concisely for cataloging. Output only the description, nothing else.",
                "max_tokens": 100,
                "temperature": 0.3,
            }
        )
        et = time.time()

        description = result.get("output", "").strip()
        print(f"DEBUG - Image described in {et - st:.2f}s: {description[:100]}...")

        return description if description else f"Generated from: {prompt_used}"

    except Exception as e:
        print(f"DEBUG - Image description failed: {e}")
        return f"Generated from: {prompt_used}"


def _extract_reasoning(text: str) -> Optional[str]:
    """Extract the [REASONING] section from LLM response."""
    # Look for [REASONING] block
    reasoning_pattern = r'\[REASONING\](.*?)(?=```json|\[ACTION\]|$)'
    matches = re.findall(reasoning_pattern, text, re.DOTALL | re.IGNORECASE)

    if matches:
        reasoning = matches[0].strip()
        print(f"DEBUG - Extracted reasoning: {reasoning[:200]}...")
        return reasoning

    return None


def _parse_tool_call(text: str) -> Optional[dict]:
    """Parse a tool call from the LLM response."""
    print(f"\n{'='*80}")
    print(f"DEBUG - Parsing tool call from LLM response...")
    print(f"DEBUG - Response text length: {len(text)} chars")
    print(f"DEBUG - Response preview: {text[:500]}...")
    print(f"{'='*80}\n")

    # Look for JSON blocks
    json_pattern = r'```json\s*(\{[^`]+\})\s*```'
    matches = re.findall(json_pattern, text, re.DOTALL)

    if matches:
        print(f"DEBUG - Found {len(matches)} JSON code block(s)")
        try:
            tool_call = json.loads(matches[-1])
            if "tool" in tool_call and "args" in tool_call:
                print(f"DEBUG - Successfully parsed tool call: {json.dumps(tool_call, indent=2)}")
                return tool_call
            else:
                print(f"DEBUG - JSON found but missing 'tool' or 'args': {tool_call}")
        except json.JSONDecodeError as e:
            print(f"DEBUG - JSON decode error in code block: {e}")
            print(f"DEBUG - Raw JSON string: {matches[-1][:200]}...")

    # Try finding raw JSON
    raw_pattern = r'\{"tool":\s*"[^"]+",\s*"args":\s*\{[^}]+\}\}'
    matches = re.findall(raw_pattern, text)

    if matches:
        print(f"DEBUG - Found {len(matches)} raw JSON match(es)")
        try:
            tool_call = json.loads(matches[-1])
            print(f"DEBUG - Successfully parsed raw tool call: {json.dumps(tool_call, indent=2)}")
            return tool_call
        except json.JSONDecodeError as e:
            print(f"DEBUG - JSON decode error in raw match: {e}")

    print(f"DEBUG - No tool call found in response")
    return None


def _enrich_prompt(
    prompt: str,
    tool_name: str,
    model_name: str = "google/gemini-2.5-flash"
) -> str:
    """Enrich a user prompt with more detail and style for better generation.

    Uses a fast LLM call to enhance simple prompts into rich, detailed ones.
    """
    if not prompt or len(prompt) > 500:
        # Skip enrichment for empty or already detailed prompts
        return prompt

    print(f"DEBUG - Enriching prompt: '{prompt[:100]}...'")

    if tool_name == "generate_images":
        system = "You enhance image generation prompts. Output ONLY the enhanced prompt, nothing else."
        enrichment_prompt = f"""Enhance this image prompt to be more vivid and detailed for AI image generation.

Original: "{prompt}"

Add:
- Specific visual details (lighting, colors, composition)
- Art style or medium (digital art, oil painting, photography, anime, etc.)
- Mood and atmosphere
- Quality keywords (highly detailed, 4k, masterpiece)

Keep it concise (under 100 words). Output ONLY the enhanced prompt."""

#     elif tool_name == "edit_images":
#         system = "You enhance image editing prompts. Output ONLY the enhanced prompt, nothing else."
#         enrichment_prompt = f"""Enhance this image editing prompt to be more specific for AI image editing.

# Original: "{prompt}"

# Make it more specific about:
# - What exactly to change/add/remove
# - How to preserve existing style and quality and content.
# - Specific visual details for the edit

# Keep it concise (under 50 words). Output ONLY the enhanced prompt."""

    elif tool_name == "generate_video":
        system = "You enhance video generation prompts. Output ONLY the enhanced prompt, nothing else."
        enrichment_prompt = f"""Enhance this video prompt to be more cinematic and detailed.

Original: "{prompt}"

Add:
- Camera movement (pan, zoom, tracking shot)
- Motion details (how things move, speed)
- Atmosphere and mood
- Cinematic quality keywords

Keep it concise (under 80 words). Output ONLY the enhanced prompt."""
    else:
        return prompt

    try:
        st = time.time()
        result = fal_client.subscribe(
            "openrouter/router",  # Non-vision endpoint for text-only
            arguments={
                "model": model_name,
                "prompt": enrichment_prompt,
                "system_prompt": system,
                "max_tokens": 200,
                "temperature": 0.7,
            }
        )
        et = time.time()

        enhanced = result.get("output", "").strip()

        # Clean up any quotes or prefixes the LLM might add
        enhanced = enhanced.strip('"\'')
        if enhanced.lower().startswith("enhanced:"):
            enhanced = enhanced[9:].strip()
        if enhanced.lower().startswith("prompt:"):
            enhanced = enhanced[7:].strip()

        print(f"DEBUG - Enriched prompt in {et - st:.2f}s: '{enhanced[:100]}...'")

        return enhanced if enhanced else prompt

    except Exception as e:
        print(f"DEBUG - Prompt enrichment failed: {e}")
        return prompt


def _execute_tool(tool_name: str, args: dict, model_name: str = "google/gemini-2.5-flash") -> tuple[str, str]:
    """Execute a tool and return the result.

    Returns:
        tuple: (result, enhanced_prompt) - enhanced_prompt is the enriched prompt if applicable
    """
    print(f"\n{'='*80}")
    print(f"DEBUG - Executing Tool: {tool_name}")
    print(f"DEBUG - Tool Args: {json.dumps(args, indent=2)}")
    print(f"{'='*80}\n")

    tools = {
        "generate_images": generate_images,
        "edit_images": edit_images,
        "generate_video": generate_video,
    }

    if tool_name not in tools:
        error_msg = f"Error: Unknown tool '{tool_name}'. Available tools: {list(tools.keys())}"
        print(f"DEBUG - Tool execution error: {error_msg}")
        return error_msg, ""

    # Enrich the prompt before execution
    original_prompt = args.get("prompt", "")
    enhanced_prompt = ""

    if original_prompt and tool_name in ["generate_images", "edit_images", "generate_video"]:
        enhanced_prompt = _enrich_prompt(original_prompt, tool_name, model_name)
        if enhanced_prompt != original_prompt:
            args = dict(args)  # Copy to avoid mutating original
            args["prompt"] = enhanced_prompt

    try:
        st = time.time()
        tool = tools[tool_name]
        print(f"DEBUG - Invoking tool.invoke() for {tool_name}...")
        result = tool.invoke(args)
        et = time.time()
        print(f"DEBUG - Tool {tool_name} completed in {et - st:.2f}s")
        print(f"DEBUG - Tool result: {result[:500] if len(result) > 500 else result}")
        return result, enhanced_prompt
    except Exception as e:
        print(f"DEBUG - Tool execution EXCEPTION: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return f"Error executing {tool_name}: {str(e)}", enhanced_prompt


@dataclass
class ContentAgent:
    """LangGraph-based content generation agent using fal.ai LLM."""

    fal_model_name: str = "google/gemini-2.5-flash"
    temperature: float = 0.7
    max_tokens: int = 4096
    _graph: Optional[StateGraph] = field(default=None, init=False)
    _memory: Optional[MemorySaver] = field(default=None, init=False)
    _llm: Optional[FalAILLM] = field(default=None, init=False)

    def __post_init__(self):
        """Initialize the agent."""
        self._memory = MemorySaver()
        self._llm = FalAILLM(
            fal_model_name=self.fal_model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        self._graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph."""
        llm = self._llm
        model_name = self.fal_model_name  # Capture for visual analysis

        def agent_node(state: AgentState) -> dict:
            """Process messages and generate response."""
            print(f"\n{'='*80}")
            print(f"DEBUG - AGENT NODE ENTERED")
            print(f"DEBUG - Number of messages in state: {len(state['messages'])}")
            print(f"DEBUG - Generated content so far: {state.get('generated_content', [])}")
            print(f"DEBUG - Generation history: {len(state.get('generation_history', []))} batches")
            print(f"DEBUG - Pending tool call: {state.get('pending_tool_call')}")
            print(f"{'='*80}\n")

            messages = list(state["messages"])

            # Add system prompt if not present
            if not messages or not isinstance(messages[0], SystemMessage):
                messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
                print(f"DEBUG - Added system prompt to messages")

            # Extract user's latest message
            user_message = ""
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage):
                    if isinstance(msg.content, str):
                        user_message = msg.content
                    elif isinstance(msg.content, list):
                        for item in msg.content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                user_message = item.get("text", "")
                                break
                    break

            # Build enhanced prompt with settings and context
            history = state.get("generation_history", [])
            settings = state.get("settings", {})
            messages_to_emit = []

            # Build settings block
            mode = settings.get("mode", "fast")
            aspect_ratio = settings.get("aspectRatio", "square")
            settings_block = f"""[Settings]
Mode: {mode} | Aspect Ratio: {aspect_ratio}
Default: Generate 4 variations unless user specifies otherwise."""

            if history:
                # Smart image lookup: find relevant images based on user message
                relevant_images, batch_metadata = _find_relevant_images(
                    user_message, history, max_images=4
                )

                # Check if visual context analysis is needed
                needs_context = _needs_visual_context(user_message) and relevant_images
                print(f"DEBUG - User message: '{user_message[:100]}...'")
                print(f"DEBUG - Needs visual context: {needs_context}")

                # Build image paths reference (all images for agent context)
                image_paths_ref = ""
                if history:
                    paths_list = []
                    for i, record in enumerate(history, 1):
                        for path in record["paths"]:
                            paths_list.append(f"  - Batch {i} Image: {path}")
                    image_paths_ref = f"\n\n[Image File Paths]\n" + "\n".join(paths_list)

                if needs_context:
                    # Format visual context from cached descriptions (no API call!)
                    visual_context = _format_visual_context(batch_metadata, user_message)

                    # Emit context message for UI
                    messages_to_emit.append(VisualAnalysisMessage(
                        content=f"📋 Found {len(relevant_images)} relevant image(s) from history"
                    ))

                    # Build enhanced prompt with cached context
                    context_parts = [
                        f"User request: {user_message}",
                        "",
                        settings_block,
                        "",
                        "[VISUAL CONTEXT - from cached descriptions]",
                        visual_context,
                        "",
                        "Use the image paths above for editing. Preserve the described style."
                    ]
                    context_text = "\n".join(context_parts)
                else:
                    # Include settings and history without visual analysis
                    context_parts = [
                        f"{user_message}",
                        "",
                        settings_block
                    ]
                    if image_paths_ref:
                        context_parts.append(image_paths_ref)
                    context_text = "\n".join(context_parts)

                # Enhance the last human message with context
                for i in range(len(messages) - 1, -1, -1):
                    if isinstance(messages[i], HumanMessage):
                        messages[i] = HumanMessage(content=context_text)
                        print(f"DEBUG - Enhanced prompt with context (visual analysis: {needs_context})")
                        break
            else:
                # First message - just add settings
                for i in range(len(messages) - 1, -1, -1):
                    if isinstance(messages[i], HumanMessage):
                        original_text = messages[i].content if isinstance(messages[i].content, str) else ""
                        if isinstance(messages[i].content, list):
                            for item in messages[i].content:
                                if isinstance(item, dict) and item.get("type") == "text":
                                    original_text = item.get("text", "")
                                    break

                        enhanced = f"{original_text}\n\n{settings_block}"
                        messages[i] = HumanMessage(content=enhanced)
                        break

            # Log last message being sent
            last_msg = messages[-1]
            if isinstance(last_msg.content, str):
                print(f"DEBUG - Last message (to LLM): {last_msg.content[:200]}...")
            else:
                print(f"DEBUG - Last message is multimodal with {len(last_msg.content)} parts")

            # Generate response
            print(f"DEBUG - Calling LLM.invoke() with {len(messages)} messages...")
            st = time.time()
            response = llm.invoke(messages)
            et = time.time()
            response_text = response.content
            print(f"DEBUG - LLM response received in {et - st:.2f}s")
            print(f"DEBUG - LLM response length: {len(response_text)} chars")
            print(f"DEBUG - LLM response:\n{response_text}")

            # Extract reasoning if present
            reasoning = _extract_reasoning(response_text)

            # Check for tool call in response
            tool_call = _parse_tool_call(response_text)

            if tool_call:
                print(f"DEBUG - Tool call detected: {tool_call['tool']}")
            else:
                print(f"DEBUG - No tool call in response (conversational reply)")

            # Build messages to return (including visual analysis + reasoning + response)
            all_messages = list(messages_to_emit)  # Start with visual analysis messages

            # Add reasoning message for UI dropdown
            if reasoning:
                all_messages.append(ReasoningMessage(
                    content=reasoning
                ))
                print(f"DEBUG - Added reasoning message")

            # Add the full response (includes both reasoning and tool call)
            all_messages.append(AIMessage(content=response_text))

            return {
                "messages": all_messages,
                "pending_tool_call": tool_call,
            }

        def should_execute_tool(state: AgentState) -> str:
            """Check if we need to execute a tool."""
            pending = state.get("pending_tool_call")
            if pending:
                print(f"DEBUG - should_execute_tool: YES -> routing to 'execute_tool' node")
                return "execute_tool"
            print(f"DEBUG - should_execute_tool: NO -> routing to END")
            return END

        def execute_tool_node(state: AgentState) -> dict:
            """Execute the pending tool call."""
            print(f"\n{'='*80}")
            print(f"DEBUG - EXECUTE TOOL NODE ENTERED")
            print(f"{'='*80}\n")

            tool_call = state.get("pending_tool_call")
            if not tool_call:
                print(f"DEBUG - No pending tool call, returning early")
                return {"pending_tool_call": None}

            tool_name = tool_call.get("tool")
            args = tool_call.get("args", {})
            original_prompt = args.get("prompt", "")

            print(f"DEBUG - About to execute tool: {tool_name}")
            print(f"DEBUG - With args: {json.dumps(args, indent=2)}")

            # Execute the tool (with prompt enrichment)
            result, enhanced_prompt = _execute_tool(tool_name, args, model_name)

            print(f"DEBUG - Tool execution finished")
            print(f"DEBUG - Result preview: {result[:300]}..." if len(result) > 300 else f"DEBUG - Result: {result}")

            # Extract any generated content paths
            new_image_paths = _extract_image_paths(result)
            new_video_paths = _extract_video_paths(result)
            new_content = new_image_paths + new_video_paths

            print(f"DEBUG - Extracted {len(new_content)} new content paths: {new_content}")

            existing_content = state.get("generated_content", [])
            existing_history = state.get("generation_history", [])

            # Create a generation record with description
            new_history = list(existing_history)
            prompt_for_record = enhanced_prompt if enhanced_prompt else original_prompt

            if new_content:
                content_type = "video" if new_video_paths else "image"

                # Describe the generated images (cache for future reference)
                description = ""
                if new_image_paths:
                    description = _describe_generated_images(
                        new_image_paths, prompt_for_record, model_name
                    )

                new_record: GenerationRecord = {
                    "prompt": prompt_for_record,
                    "paths": new_content,
                    "type": content_type,
                    "description": description,
                }
                new_history.append(new_record)
                print(f"DEBUG - Added generation record: {content_type} with description '{description[:50]}...'" if description else f"DEBUG - Added generation record: {content_type}")

            # Build messages to return
            messages_to_return = []

            # Add prompt enrichment message if prompt was enhanced
            if enhanced_prompt and enhanced_prompt != original_prompt:
                enrichment_msg = f"✨ Enhanced prompt: {enhanced_prompt}"
                messages_to_return.append(ReasoningMessage(content=enrichment_msg))
                print(f"DEBUG - Added prompt enrichment message")

            # Add content reflection message
            if new_content:
                content_count = len(new_content)
                content_type_str = "video" if new_video_paths else "image"
                reflection = f"✅ Generated {content_count} {content_type_str}(s)"
                messages_to_return.append(ReasoningMessage(content=reflection))
                print(f"DEBUG - Added content reflection message")

            # Return tool result as AIMessage (so it displays to user)
            result_message = AIMessage(content=result)
            messages_to_return.append(result_message)

            print(f"DEBUG - Returning from execute_tool_node, going to END")

            return {
                "messages": messages_to_return,
                "generated_content": existing_content + new_content,
                "generation_history": new_history,
                "pending_tool_call": None,
            }

        # Build the graph
        graph = StateGraph(AgentState)

        graph.add_node("agent", agent_node)
        graph.add_node("execute_tool", execute_tool_node)

        graph.set_entry_point("agent")

        graph.add_conditional_edges(
            "agent",
            should_execute_tool,
            {"execute_tool": "execute_tool", END: END}
        )

        # After tool execution, END the graph (don't loop back to agent)
        graph.add_edge("execute_tool", END)

        return graph.compile(checkpointer=self._memory)

    def _prepare_multimodal_message(
        self,
        text: str,
        image_paths: Optional[List[str]] = None
    ) -> HumanMessage:
        """Prepare a multimodal message with text and images."""
        if not image_paths:
            return HumanMessage(content=text)

        content = []

        # Add images first
        for path in image_paths:
            base64_url = _image_to_base64(path)
            if base64_url:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": base64_url}
                })

        # Add text
        content.append({"type": "text", "text": text})

        return HumanMessage(content=content)

    def invoke(
        self,
        message: str,
        image_paths: Optional[List[str]] = None,
        thread_id: str = "default",
        settings: Optional[dict] = None
    ) -> dict:
        """Invoke the agent with a message and optional images."""
        human_message = self._prepare_multimodal_message(message, image_paths)

        config = {"configurable": {"thread_id": thread_id}}

        # IMPORTANT: Only provide new inputs, not full state
        # The checkpointer will merge with existing state
        # Don't override generated_content or generation_history
        initial_state = {
            "messages": [human_message],
            "settings": settings or {},
        }

        result = self._graph.invoke(initial_state, config)
        return result

    def stream(
        self,
        message: str,
        image_paths: Optional[List[str]] = None,
        thread_id: str = "default",
        settings: Optional[dict] = None
    ):
        """Stream the agent's response."""
        human_message = self._prepare_multimodal_message(message, image_paths)

        config = {"configurable": {"thread_id": thread_id}}

        # IMPORTANT: Only provide new inputs, not full state
        # The checkpointer will merge with existing state
        initial_state = {
            "messages": [human_message],
            "settings": settings or {},
        }

        for event in self._graph.stream(initial_state, config, stream_mode="values"):
            yield event

    def get_conversation_history(self, thread_id: str = "default") -> List[BaseMessage]:
        """Get the conversation history for a thread."""
        config = {"configurable": {"thread_id": thread_id}}
        state = self._graph.get_state(config)
        if state and state.values:
            return state.values.get("messages", [])
        return []

    def get_generated_content(self, thread_id: str = "default") -> List[str]:
        """Get all generated content paths for a thread."""
        config = {"configurable": {"thread_id": thread_id}}
        state = self._graph.get_state(config)
        if state and state.values:
            return state.values.get("generated_content", [])
        return []

    def get_generation_history(self, thread_id: str = "default") -> List[GenerationRecord]:
        """Get the structured generation history for a thread."""
        config = {"configurable": {"thread_id": thread_id}}
        state = self._graph.get_state(config)
        if state and state.values:
            return state.values.get("generation_history", [])
        return []


def create_agent(
    fal_model_name: str = "google/gemini-2.5-flash",
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> ContentAgent:
    """Create a content generation agent.

    Args:
        fal_model_name: The LLM to use via fal.ai's any-llm endpoint.
            Examples: "google/gemini-2.5-flash", "anthropic/claude-3.5-sonnet",
                      "openai/gpt-4o", "meta-llama/llama-3.1-70b-instruct"
        temperature: Temperature for LLM responses
        max_tokens: Maximum tokens for LLM responses

    Returns:
        A configured ContentAgent instance
    """
    return ContentAgent(
        fal_model_name=fal_model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
