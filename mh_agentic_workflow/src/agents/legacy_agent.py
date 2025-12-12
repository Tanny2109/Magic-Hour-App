"""Legacy FalAI client implementation (Option 2 from guide)"""
import io
import json
import re
from typing import List, Tuple, Optional, Dict, Any

import fal_client
import requests
from PIL import Image


class FalAIClient:
    """Legacy client for fal.ai that uses fal-ai/any-llm for intent detection"""
    
    def __init__(self):
        self.available_models = {
            "chat-llm": "fal-ai/any-llm",
            "nano-banana": "fal-ai/nano-banana",
        }

        self.conversation_history = []
        self.uploaded_image = None
        self.user_preferences = {
            "default_image_model": "nano-banana",
            "default_video_model": "luma-dream-machine",
            "default_image_size": {"width": 720, "height": 720},
            "default_video_duration": 5
        }

        # System prompt for the LLM to understand its role
        self.system_prompt = """You are an intelligent assistant that helps users with image generation, image editing, and video creation using fal.ai models.

Your job is to:
1. Understand what the user wants to do
2. Determine the appropriate operation and model
3. Extract or suggest parameters
4. Provide a structured response

AVAILABLE OPERATIONS:
- "generate_image": Create new images from text prompts
- "edit_image": Modify existing uploaded images
- "generate_video": Create videos from text prompts
- "chat": General conversation, questions, or explanations
- "change_settings": Modify default parameters or models

AVAILABLE MODELS:
Image Generation: nano-banana
Video Generation: luma-dream-machine

Always respond with a JSON object containing:
{
  "operation": "generate_image|edit_image|generate_video|chat|change_settings",
  "model": "recommended_model_name",
  "prompt": "cleaned_up_prompt_for_generation",
  "parameters": {
    "width": 1024,
    "height": 1024,
    "num_images": 1,
    "guidance_scale": 7.5,
    "duration": 3
  },
  "explanation": "Brief explanation of what you'll do",
  "chat_response": "Conversational response to the user"
}

Be intelligent about model selection and parameter extraction."""

    def query_llm(self, user_message: str) -> Dict[str, Any]:
        """Query the any-llm model to understand user intent
        
        Args:
            user_message: User's input message
            
        Returns:
            Dictionary containing operation, model, prompt, parameters, etc.
        """
        try:
            handler = fal_client.submit(
                self.available_models["chat-llm"],
                arguments={
                    "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
                    "system_prompt": self.system_prompt + f"\nUser's current preferences: {json.dumps(self.user_preferences)}",
                    "prompt": f"Conversation context:{user_message}\n\nPlease analyze this request and respond in JSON.",
                    "max_tokens": 1000,
                    "temperature": 0.9,
                    "top_p": 0.9
                }
            )

            result = handler.get()
            print(result)

            # Handle the response format
            response_text = ""
            if "output" in result:
                response_text = result["output"]
            elif "choices" in result and result["choices"]:
                response_text = result["choices"][0]["message"]["content"]
            else:
                response_text = str(result)

            # Try to extract and parse JSON from the response
            try:
                # Look for JSON in the response text
                json_match = re.search(
                    r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL
                )
                if json_match:
                    json_str = json_match.group(0)

                    # Clean up the JSON string - fix common formatting issues
                    json_str = json_str.replace('width": ,', 'width": 1024,')
                    json_str = json_str.replace('height": ,', 'height": 1024,')
                    json_str = json_str.replace('num_images": ,', 'num_images": 1,')
                    json_str = json_str.replace('guidance_scale": .', 'guidance_scale": 7.5')
                    json_str = json_str.replace('duration": ,', 'duration": 3,')

                    parsed_json = json.loads(json_str)

                    # Validate and fix model name
                    requested_model = parsed_json.get("model", "")
                    if requested_model not in self.available_models:
                        operation = parsed_json.get("operation", "")
                        if operation == "generate_image":
                            parsed_json["model"] = self.user_preferences["default_image_model"]
                        elif operation == "edit_image":
                            parsed_json["model"] = "flux-inpainting"
                        elif operation == "generate_video":
                            parsed_json["model"] = self.user_preferences["default_video_model"]

                    # Ensure parameters have default values
                    if "parameters" in parsed_json:
                        params = parsed_json["parameters"]
                        params.setdefault("width", self.user_preferences["default_image_size"]["width"])
                        params.setdefault("height", self.user_preferences["default_image_size"]["height"])
                        params.setdefault("num_images", 1)
                        params.setdefault("guidance_scale", 7.5)
                        params.setdefault("duration", self.user_preferences["default_video_duration"])

                    return parsed_json

                # If no JSON found, extract key information manually
                return self._fallback_parse(user_message, response_text)

            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                print(f"Raw response: {response_text}")
                return self._fallback_parse(user_message, response_text)

        except Exception as e:
            print(f"LLM query error: {e}")
            return {
                "operation": "chat",
                "chat_response": f"I encountered an error: {str(e)}. Let me try to help you anyway!",
                "explanation": f"LLM Error: {str(e)}"
            }

    def _fallback_parse(self, user_message: str, response_text: str) -> Dict[str, Any]:
        """Fallback parsing when JSON extraction fails"""
        lines = response_text.lower().split('\n')
        operation = "chat"

        for line in lines:
            if "generate" in line and ("image" in line or "picture" in line or "photo" in line):
                operation = "generate_image"
                break
            elif "edit" in line or "modify" in line or "transform" in line:
                operation = "edit_image"
                break
            elif "video" in line or "animate" in line:
                operation = "generate_video"
                break

        return {
            "operation": operation,
            "model": self.user_preferences["default_image_model"],
            "prompt": user_message,
            "parameters": {
                "width": self.user_preferences["default_image_size"]["width"],
                "height": self.user_preferences["default_image_size"]["height"],
                "num_images": 1,
                "guidance_scale": 7.5,
            },
            "explanation": f"Processing your request to {operation.replace('_', ' ')}",
            "chat_response": "I'll process your request now!"
        }

    def generate_image(
        self, prompt: str, model: str, parameters: Dict[str, Any]
    ) -> Tuple[Optional[Image.Image], str]:
        """Generate images using specified model and parameters
        
        Args:
            prompt: Text description for image generation
            model: Model identifier
            parameters: Generation parameters
            
        Returns:
            Tuple of (generated_image, status_message)
        """
        try:
            args = {
                "prompt": prompt,
                "image_size": {
                    "width": parameters.get("width", 1024),
                    "height": parameters.get("height", 1024)
                },
                "num_images": parameters.get("num_images", 1),
                "guidance_scale": parameters.get("guidance_scale", 7.5),
                "num_inference_steps": parameters.get("num_inference_steps", 28)
            }

            handler = fal_client.submit(self.available_models[model], arguments=args)
            result = handler.get()

            images = []
            if "images" in result:
                for img_data in result["images"]:
                    img_url = img_data.get("url")
                    if img_url:
                        response = requests.get(img_url)
                        img = Image.open(io.BytesIO(response.content))
                        images.append(img)

            print(f"Generated images: {result}")
            return images[0] if images else None, "success"

        except Exception as e:
            return None, f"Error generating images: {str(e)}"

    def update_preferences(self, new_preferences: Dict[str, Any]) -> str:
        """Update user preferences
        
        Args:
            new_preferences: Dictionary of preferences to update
            
        Returns:
            Status message
        """
        self.user_preferences.update(new_preferences)
        return f"Updated preferences: {json.dumps(new_preferences, indent=2)}"

    def process_message(
        self, user_message: str, history: List
    ) -> Tuple[List[Optional[Image.Image]], str]:
        """Process user message and return response
        
        Args:
            user_message: User's input message
            history: Chat history
            
        Returns:
            Tuple of (images_list, response_text)
        """
        self.conversation_history.append(("User", user_message))

        llm_response = self.query_llm(user_message)

        operation = llm_response.get("operation", "chat")
        chat_response = llm_response.get("chat_response", "")
        explanation = llm_response.get("explanation", "")

        response_text = ""
        images = []

        try:
            if operation == "generate_image":
                model = llm_response.get(
                    "model", self.user_preferences["default_image_model"]
                )
                prompt = llm_response.get("prompt", user_message)
                parameters = llm_response.get("parameters", {})

                image, status = self.generate_image(prompt, model, parameters)

                if status == "success" and image:
                    images = [image]
                    response_text = (
                        f"**{explanation}**\n\n"
                        f"*Generated using {model}*\n"
                        f"*Prompt: \"{prompt}\"*\n\n"
                        f"{chat_response}"
                    )
                else:
                    response_text = f"{status}\n\n{chat_response}"

            else:  # chat operation or unsupported operations
                response_text = chat_response

        except Exception as e:
            response_text = f"**Error:** {str(e)}\n\n{chat_response}"

        # Add assistant response to history
        self.conversation_history.append(("Assistant", response_text))

        return images, response_text
