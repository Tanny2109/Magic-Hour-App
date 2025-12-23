"""
FalAI LLM implementation for LangChain using OpenRouter Vision API.

This module provides a custom ChatModel class that wraps fal.ai's OpenRouter
Vision API to work with LangChain and LangGraph agents.
"""

import os
import json
import base64
import logging
import time
from typing import Any, Iterator, List, Optional

import fal_client
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field

logger = logging.getLogger(__name__)


class FalAILLM(BaseChatModel):
    """
    A LangChain ChatModel that uses fal.ai's OpenRouter Vision API.

    This allows using various LLMs (Gemini, Claude, GPT, etc.) through fal.ai's
    infrastructure with multimodal support.

    Example:
        ```python
        from src.models import FalAILLM

        llm = FalAILLM(
            fal_model_name="google/gemini-2.5-flash",
            temperature=0.7,
        )

        response = llm.invoke("Hello, how are you?")
        ```
    """

    fal_api: str = Field(default="openrouter/router")
    fal_model_name: str = Field(default="google/gemini-3-flash-preview")
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=4096)
    api_key: Optional[str] = Field(default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Configure fal_client
        if self.api_key:
            fal_client.api_key = self.api_key
        elif os.getenv("FAL_KEY"):
            fal_client.api_key = os.getenv("FAL_KEY")

        logger.info(f"Initialized FalAILLM with model: {self.fal_model_name}")

    @property
    def _llm_type(self) -> str:
        return "fal-ai-openrouter"

    @property
    def _identifying_params(self) -> dict:
        return {
            "fal_api": self.fal_api,
            "fal_model_name": self.fal_model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def _extract_content_and_images(
        self,
        messages: List[BaseMessage]
    ) -> tuple[str, Optional[str], List[str]]:
        """
        Extract prompt text, system prompt, and image URLs from messages.

        Returns:
            tuple: (prompt_text, system_prompt, image_urls)
        """
        system_prompt = None
        prompt_parts = []
        image_urls = []

        for msg in messages:
            if isinstance(msg, SystemMessage):
                # Extract system prompt
                if isinstance(msg.content, str):
                    if system_prompt:
                        system_prompt += "\n" + msg.content
                    else:
                        system_prompt = msg.content

            elif isinstance(msg, HumanMessage):
                content = msg.content

                if isinstance(content, str):
                    prompt_parts.append(f"User: {content}")

                elif isinstance(content, list):
                    # Multimodal content
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict):
                            item_type = item.get("type")

                            if item_type == "text":
                                text_parts.append(item.get("text", ""))

                            elif item_type == "image_url":
                                image_url_data = item.get("image_url", {})
                                if isinstance(image_url_data, dict):
                                    url = image_url_data.get("url", "")
                                else:
                                    url = str(image_url_data)

                                if url:
                                    # Handle file paths - upload to fal.ai
                                    if url.startswith("/") or url.startswith("file://"):
                                        file_path = url.replace("file://", "")
                                        if os.path.exists(file_path):
                                            try:
                                                with open(file_path, "rb") as f:
                                                    image_data = f.read()
                                                if file_path.lower().endswith(".png"):
                                                    content_type = "image/png"
                                                elif file_path.lower().endswith((".jpg", ".jpeg")):
                                                    content_type = "image/jpeg"
                                                else:
                                                    content_type = "image/png"
                                                uploaded_url = fal_client.upload(image_data, content_type)
                                                image_urls.append(uploaded_url)
                                            except Exception as e:
                                                logger.warning(f"Failed to upload image {file_path}: {e}")
                                    elif url.startswith("data:"):
                                        # Base64 data URL - upload to fal.ai
                                        try:
                                            # Extract base64 data
                                            header, b64_data = url.split(",", 1)
                                            image_data = base64.b64decode(b64_data)
                                            content_type = header.split(";")[0].replace("data:", "")
                                            uploaded_url = fal_client.upload(image_data, content_type)
                                            image_urls.append(uploaded_url)
                                        except Exception as e:
                                            logger.warning(f"Failed to upload base64 image: {e}")
                                    else:
                                        # Regular URL
                                        image_urls.append(url)

                        elif isinstance(item, str):
                            text_parts.append(item)

                    if text_parts:
                        prompt_parts.append(f"User: {' '.join(text_parts)}")

            elif isinstance(msg, AIMessage):
                if isinstance(msg.content, str):
                    prompt_parts.append(f"Assistant: {msg.content}")

        # Combine all prompt parts
        prompt = "\n\n".join(prompt_parts)

        return prompt, system_prompt, image_urls

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response using fal.ai OpenRouter Vision API."""
        print(f"\n{'='*80}")
        print(f"DEBUG - FalAILLM._generate() called")
        print(f"DEBUG - Model: {self.fal_model_name}")
        print(f"DEBUG - API: {self.fal_api}")
        print(f"DEBUG - Number of messages: {len(messages)}")
        print(f"{'='*80}\n")

        prompt, system_prompt, image_urls = self._extract_content_and_images(messages)

        print(f"DEBUG - Extracted prompt length: {len(prompt)} chars")
        print(f"DEBUG - System prompt present: {bool(system_prompt)}")
        print(f"DEBUG - Image URLs: {len(image_urls)}")
        if system_prompt:
            print(f"DEBUG - System prompt preview: {system_prompt[:200]}...")
        print(f"DEBUG - Prompt preview: {prompt[:300]}...")

        # Prepare fal.ai OpenRouter request
        fal_input = {
            "model": self.fal_model_name,
            "prompt": prompt,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }

        if system_prompt:
            fal_input["system_prompt"] = system_prompt

        if image_urls:
            fal_input["image_urls"] = image_urls

        # Debug the actual request
        debug_input = {k: v for k, v in fal_input.items() if k != "system_prompt"}
        debug_input["prompt"] = prompt[:200] + "..." if len(prompt) > 200 else prompt
        print(f"DEBUG - fal.ai request:")
        print(json.dumps(debug_input, indent=2, default=str))

        try:
            print(f"DEBUG - Calling fal_client.subscribe({self.fal_api})...")
            st = time.time()
            result = fal_client.subscribe(self.fal_api, arguments=fal_input)
            et = time.time()
            print(f"DEBUG - fal.ai response received in {et - st:.2f}s")

            output_text = result.get("output", "")
            print(f"DEBUG - Response output length: {len(output_text)} chars")
            print(f"DEBUG - Response: {output_text}")

            if not output_text:
                print(f"DEBUG - WARNING: Empty output from fal.ai!")
                print(f"DEBUG - Full result: {result}")

            message = AIMessage(content=output_text)
            generation = ChatGeneration(message=message)

            return ChatResult(generations=[generation])

        except Exception as e:
            print(f"DEBUG - ERROR calling fal.ai API: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            logger.error(f"Error calling fal.ai API: {e}")
            raise

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream a response using fal.ai OpenRouter Vision API."""
        prompt, system_prompt, image_urls = self._extract_content_and_images(messages)

        fal_input = {
            "model": self.fal_model_name,
            "prompt": prompt,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }

        if system_prompt:
            fal_input["system_prompt"] = system_prompt

        if image_urls:
            fal_input["image_urls"] = image_urls

        logger.debug(f"Streaming from fal.ai with model: {self.fal_model_name}")

        try:
            stream = fal_client.stream(self.fal_api, arguments=fal_input)

            for event in stream:
                if isinstance(event, dict):
                    content = event.get("output", "")
                else:
                    content = str(event)

                if content:
                    chunk = ChatGenerationChunk(
                        message=AIMessageChunk(content=content)
                    )

                    if run_manager:
                        run_manager.on_llm_new_token(content)

                    yield chunk

        except Exception as e:
            logger.error(f"Error streaming from fal.ai API: {e}")
            raise

    def bind_tools(self, tools: List[Any], **kwargs) -> "FalAILLM":
        """
        Bind tools to the LLM for function calling.

        Note: OpenRouter Vision API doesn't natively support tool calling,
        so we handle this through prompt engineering in the agent.
        """
        return self
