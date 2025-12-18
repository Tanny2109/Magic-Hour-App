"""Custom fal.ai tools for smolagents framework"""
import os
import tempfile
import json
from io import BytesIO
from time import time
from typing import Optional

import fal_client
import requests
from PIL import Image
from smolagents import Tool

from requests.adapters import HTTPAdapter, Retry
import os
import requests
import sys
from dotenv import load_dotenv

load_dotenv()

prodia_token = os.getenv('PRODIA_KEY')
prodia_url = 'https://inference.prodia.com/v2/job'

session = requests.Session()
retries = Retry(allowed_methods=None, status_forcelist=Retry.RETRY_AFTER_STATUS_CODES)
session.mount('http://', HTTPAdapter(max_retries=retries))
session.mount('https://', HTTPAdapter(max_retries=retries))
session.headers.update({'Authorization': f"Bearer {prodia_token}"})

headers = {
    'Accept': "image/jpeg",
    'Content-Type': 'application/json',
}

class FalImageGenerationTool(Tool):
    """Tool for generating images using fal.ai models"""

    name = "fal_image_generation"
    description = """
    Generates images using fal.ai models. Model is selected based on performance mode and the type of image being generated.
    Use this tool when the user wants to create or generate new images from text descriptions. Select based on user intent and pro vs fast model
    based on that intent.
    
    Anime: pro=seedream4, fast=flux_krea
    Cartoon and Illustrations: pro=imagen4_ultra, fast=flux_krea
    General and Photorealistic: pro=seedream4, fast=flux_krea
    Graphic Design and Digital Rendering: pro=seedream4, fast=flux_krea
    Traditional Art: pro=seedream4, fast=flux_krea
    Vintage and Retro: pro=seedream4, fast=flux_krea
    Commercial(ads, product shots): pro=imagen4_ultra, fast=flux_krea
    Fantasy and Mythical: pro=seedream4, fast=flux_krea
    Futuristic and Sci-Fi: pro=seedream4, fast=flux_krea
    Nature and Landscapes: pro=seedream4, fast=flux_krea
    People- Groups and Activities: pro=seedream4, fast=flux_krea
    People- Portraits: pro=seedream4, fast=flux_krea
    Physical Spaces: pro=seedream4, fast=flux_krea
    Text and Typography: pro=imagen4_ultra, fast=flux_krea
    UI/UX: pro=imagen4_ultra, fast=flux_krea

    CRITICAL: This tool returns LOCAL FILE PATHS (e.g., /var/folders/.../tmp*.png).
    You MUST use these EXACT paths in final_answer(). 
    DO NOT create or use any https:// URLs - they are not valid and will cause errors.
    ONLY use the file paths returned by this tool.
    """
    inputs = {
        "prompt": {
            "type": "string",
            "description": "The text description of the image to generate"
        },
        "model": {
            "type": "string",
            "description": "model identifier (nano-banana, seedream4, imagen4_ultra, flux_krea). Default: flux_krea",
            "nullable": False
        },
        "mode":{
            "type": "string",
            "description": "Performance mode (pro or fast). Default: fast",
            "nullable": False
        },
        "image_type":{
            "type": "string",
            "description": "Type of image to generate (e.g., anime, photorealistic, cartoon, graphic design, etc.). Default: general",
            "nullable": False
        },
        "width": {
            "type": "integer",
            "description": "Image width in pixels. Infer this value based on aspect ratio. Default: 720",
            "nullable": True
        },
        "height": {
            "type": "integer",
            "description": "Image height in pixels. Infer this value based on aspect ratio. Default: 720",
            "nullable": True
        },
        "num_images": {
            "type": "integer",
            "description": "Number of images to generate (1-4). Default: 1",
            "nullable": True
        }
    }
    output_type = "string"

    def __init__(self):
        super().__init__()
        self.models = {
            "nano-banana": "fal-ai/nano-banana",
            "seedream4": "fal-ai/bytedance/seedream/v4/text-to-image",
            "imagen4_ultra": "fal-ai/imagen4/preview/ultra",
            "flux-krea": "fal-ai/flux/krea",
            "flux-schnell": "fal-ai/flux/krea" # i dunno but agent refers to schnell instead of krea at time..
        }

    def forward(
        self,
        prompt: str,
        model: str = "flux-krea",
        width: int = 720,
        height: int = 720,
        num_images: int = 1,
        mode: str = "fast",
        image_type: str = "general"
    ) -> str:
        """Generate images using fal.ai"""
        try:
            image_paths = []
            if mode=="fast" or model == "flux-schnell" or model == "flux_krea" or model == "flux-krea":
                import concurrent.futures

                # Print debug info
                debug_info = {
                    "tool": "FalImageGenerationTool",
                    "mode": mode,
                    "num_images": num_images
                }
                print(f"\n{'='*80}")
                print(f"DEBUG - Image Generation Tool Called:")
                print(json.dumps(debug_info, indent=2))
                print(f"{'='*80}\n")

                def generate_single_image(idx):
                    job = {
                        "type": "inference.flux-fast.schnell.txt2img.v2",
                        "config": {
                            "prompt": prompt,
                            "width": width,
                            "height": height,
                            "steps": 4
                        }
                    }
                    res = session.post(prodia_url, headers=headers, json=job)
                    img = Image.open(BytesIO(res.content))
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                    img.save(temp_file.name)
                    return temp_file.name

                st = time()
                # Generate multiple images in parallel for speed
                with concurrent.futures.ThreadPoolExecutor(max_workers=num_images) as executor:
                    futures = [executor.submit(generate_single_image, i) for i in range(num_images)]
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            image_paths.append(future.result())
                        except Exception as e:
                            print(f"Error generating image: {e}")
                et = time()
                print(f"DEBUG - Generated {len(image_paths)} images in {et - st:.2f} seconds")
            else:
                # Get model from config
                model_id = self.models.get(model)

                args = {
                    "prompt": prompt,
                    "image_size": {"width": width, "height": height},
                    "num_images": num_images,
                }

                # Print debug info
                debug_info = {
                    "tool": "FalImageGenerationTool",
                    "model": model_id,
                    "model_id": model_id,
                    "arguments": args
                }
                print(f"\n{'='*80}")
                print(f"DEBUG - Image Generation Tool Called:")
                print(json.dumps(debug_info, indent=2))
                print(f"{'='*80}\n")

                st = time()
                handler = fal_client.submit(model_id, arguments=args)
                result = handler.get()
                et = time()
                print(f"DEBUG - Image generation result: {result}")
                print(f"DEBUG - Time taken for image generation: {et - st} seconds")

                # Download and save images
                if "images" in result:
                    for idx, img_data in enumerate(result["images"]):
                        img_url = img_data.get("url")
                        if img_url:
                            response = requests.get(img_url)
                            img = Image.open(BytesIO(response.content))

                            # Save to temp file
                            temp_file = tempfile.NamedTemporaryFile(
                                delete=False, suffix=".png"
                            )
                            img.save(temp_file.name)
                            image_paths.append(temp_file.name)

            result_msg = f"Generated {len(image_paths)} image(s): {', '.join(image_paths)}"
            print(result_msg)
            return result_msg

        except Exception as e:
            return f"Error generating images: {str(e)}"


class FalVideoGenerationTool(Tool):
    """Tool for generating videos using fal.ai models"""
    
    name = "fal_video_generation"
    description = """
    Generates videos using fal.ai video models.
    Use this tool when the user wants to create videos or animations from text descriptions.
    
    pro mode: veo3
    fast mode: veo3_fast

    IMPORTANT: After calling this tool and receiving the generated video path, you MUST call final_answer()
    with the video path to complete the task. Do NOT call this tool multiple times for the same request.
    """
    inputs = {
        "prompt": {
            "type": "string",
            "description": "The text description of the video to generate"
        },
        "duration": {
            "type": "integer",
            "description": "Video duration in seconds (4-8). Default: 6",
            "nullable": True
        },
        "model": {
            "type": "string",
            "description": "model identifier (veo3, veo3_fast). Default: veo3_fast",
            "nullable": False
        },
        "mode":{
            "type": "string",
            "description": "Performance mode (pro or fast). Default: fast",
            "nullable": False
        },
    }
    output_type = "string"
    
    def __init__(self):
        super().__init__()
        self.models = {
            "veo3": "fal-ai/veo3",
            "veo3_fast": "fal-ai/veo3/fast",
        }
        
    def forward(self, prompt: str, model:str="veo3_fast", mode:str="fast", duration: int = 6) -> str:
        """Generate video using fal.ai

        Args:
            prompt: Text description of the video
            duration: Video duration in seconds (3-10)

        Returns:
            String with local video file path or error message
        """
        try:
            # Get model from config
            # model_id = self.models[model]
            model_id = "fal-ai/bytedance/seedance/v1/pro/fast/text-to-video"

            args = {
                "prompt": prompt,
                "duration": min(max(duration, 6), 6),
                "resolution": "720p"
            }

            # Print debug info
            debug_info = {
                "tool": "FalVideoGenerationTool",
                "mode": mode,
                "model_id": model_id,
                "arguments": args
            }
            print(f"\n{'='*80}")
            print(f"DEBUG - Video Generation Tool Called:")
            print(json.dumps(debug_info, indent=2))
            print(f"{'='*80}\n")

            st = time()
            handler = fal_client.submit(model_id, arguments=args)
            result = handler.get()
            et = time()
            print(f"DEBUG - Video generation result: {result}")
            print(f"DEBUG - Time taken for video generation: {et - st} seconds")

            if "video" in result:
                video_url = result["video"].get("url")

                # Download video to temp file
                response = requests.get(video_url)
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                temp_file.write(response.content)
                temp_file.close()

                result_msg = f"Generated video saved to: {temp_file.name}"
                print(result_msg)
                return result_msg

            # # Return mock response for testing (uncomment API calls above when ready)
            # return f"[MOCK] Would generate {duration}s video using model '{model_id}' in {mode} mode"

        except Exception as e:
            return f"Error generating video: {str(e)}"


class FalImageEditTool(Tool):
    """Tool for editing images using fal.ai models"""
    
    name = "fal_image_edit"
    description = """
    Edits or modifies existing images using fal.ai models.
    Use this tool when the user wants to edit, modify, or transform an uploaded image.
    """
    inputs = {
        "image_path": {
            "type": "string",
            "description": "Path to the image file to edit"
        },
        "prompt": {
            "type": "string",
            "description": "Description of the edits or transformations to apply"
        },
        "strength": {
            "type": "number",
            "description": "Edit strength (0.0-1.0). Default: 0.8",
            "nullable": True
        }
    }
    output_type = "string"

    def forward(self, image_path: str, prompt: str, strength: float = 0.8) -> str:
        """Edit image using fal.ai

        Args:
            image_path: Path to the image file
            prompt: Description of edits to apply
            strength: Edit strength (0.0-1.0)

        Returns:
            String with edited image path or error message
        """
        try:
            # Get model from config
            # model_id = "fal-ai/nano-banana/edit" #13.2s
            model_id = "fal-ai/bytedance/seedream/v4/edit" #13.16s

            # Upload image to fal.ai (manual upload required)
            with open(image_path, 'rb') as f:
                image_data = f.read()
            image_url = fal_client.upload(image_data, "image/png")

            args = {
                "image_urls": [image_url],
                "prompt": prompt,
                # "strength": strength,
                "width": 1024,
                "height": 1024
            }

            # Print debug info
            debug_info = {
                "tool": "FalImageEditTool",
                "model_id": model_id,
                "arguments": {
                    "prompt": prompt,
                    "image_url": image_url[:50] + "..."
                }
            }
            print(f"\n{'='*80}")
            print(f"DEBUG - Image Edit Tool Called:")
            print(json.dumps(debug_info, indent=2))
            print(f"{'='*80}\n")

            st= time()
            handler = fal_client.submit(model_id, arguments=args)
            result = handler.get()
            et = time()
            print(f"DEBUG - Time taken for image editing: {et - st} seconds")
            print(f"DEBUG - Image editing result: {result}")

            if "images" in result and result["images"]:
                edited_url = result["images"][0].get("url")

                # Download edited image
                response = requests.get(edited_url)
                img = Image.open(BytesIO(response.content))

                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                img.save(temp_file.name)

                result_msg = f"Edited image saved to: {temp_file.name}"
                print(result_msg)
                return result_msg

            # Return mock response for testing (uncomment API calls above when ready)
            # return f"[MOCK] Would edit image with prompt '{prompt}' using model '{model_id}'"

        except Exception as e:
            return f"Error editing image: {str(e)}"
