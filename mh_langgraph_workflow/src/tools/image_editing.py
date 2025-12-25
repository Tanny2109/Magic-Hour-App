"""Image editing tool using fal.ai API."""
import os
import json
import time
import tempfile
from typing import List, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

import fal_client
import requests
from pydantic import BaseModel, Field, field_validator
from langchain_core.tools import tool

# Configure fal.ai
fal_client.api_key = os.getenv("FAL_KEY")

EDIT_MODEL = "fal-ai/bytedance/seedream/v4.5/edit"


class ImageEditInput(BaseModel):
    """Input schema for image editing."""
    image_paths: Union[str, List[str]] = Field(
        description="File path(s) of images to edit. Can be a single path, comma-separated paths, or a list of paths."
    )
    prompt: str = Field(
        description="Text description of how to edit the image(s). Be specific about the changes you want."
    )

    @field_validator('image_paths', mode='before')
    @classmethod
    def normalize_paths(cls, v):
        """Convert list to comma-separated string for consistent handling."""
        if isinstance(v, list):
            return ", ".join(str(p) for p in v)
        return v


def _upload_image_to_fal(image_path: str) -> str:
    """Upload a local image to fal.ai CDN and return the URL."""
    with open(image_path, "rb") as f:
        image_data = f.read()

    # Determine content type
    if image_path.lower().endswith(".png"):
        content_type = "image/png"
    elif image_path.lower().endswith((".jpg", ".jpeg")):
        content_type = "image/jpeg"
    elif image_path.lower().endswith(".webp"):
        content_type = "image/webp"
    else:
        content_type = "image/png"

    return fal_client.upload(image_data, content_type)


def _edit_single_image(image_path: str, prompt: str) -> str:
    """Edit a single image using fal.ai API."""
    # Upload image to fal.ai CDN
    image_url = _upload_image_to_fal(image_path)

    # API expects image_urls as a list (matching smolagents implementation)
    args = {
        "image_urls": [image_url],
        "prompt": prompt,
        "width": 1024,
        "height": 1024,
    }

    result = fal_client.submit(EDIT_MODEL, arguments=args).get()

    if not result or "images" not in result or not result["images"]:
        raise ValueError("No edited image returned by fal.ai")

    output_url = result["images"][0]["url"]

    # Download the edited image
    response = requests.get(output_url, timeout=60)
    response.raise_for_status()

    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    temp_file.write(response.content)
    temp_file.close()

    return temp_file.name


@tool(args_schema=ImageEditInput)
def edit_images(image_paths: str, prompt: str) -> str:
    """Edit existing images based on a text prompt.

    Use this tool when the user wants to modify, transform, or change existing images.
    The user must provide the file path(s) of images to edit.

    Args:
        image_paths: Comma-separated file paths of images to edit
        prompt: Description of how to edit the image(s)

    Returns:
        String containing the file paths of edited images, separated by newlines
    """
    # Parse paths
    paths = [p.strip() for p in image_paths.split(",") if p.strip()]

    # Debug logging
    debug_info = {
        "image_paths": image_paths,
        "parsed_paths": paths,
        "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
        "num_images": len(paths),
    }
    print(f"\n{'='*80}")
    print(f"DEBUG - Image Edit Tool Called:")
    print(json.dumps(debug_info, indent=2))
    print(f"{'='*80}\n")

    if not paths:
        print(f"DEBUG - ERROR: No valid image paths provided")
        return "Error: No valid image paths provided"

    # Validate paths exist
    valid_paths = []
    for path in paths:
        if os.path.exists(path):
            valid_paths.append(path)
            print(f"DEBUG - Valid path: {path}")
        else:
            print(f"DEBUG - ERROR: File not found: {path}")
            return f"Error: Image file not found: {path}"

    print(f"DEBUG - {len(valid_paths)} valid paths to edit")

    edited_paths = []
    errors = []

    st = time.time()

    if len(valid_paths) == 1:
        try:
            print(f"DEBUG - Editing single image: {valid_paths[0]}")
            edited_path = _edit_single_image(valid_paths[0], prompt)
            edited_paths.append(edited_path)
            print(f"DEBUG - Single image edited: {edited_path}")
        except Exception as e:
            print(f"DEBUG - Single image edit EXCEPTION: {type(e).__name__}: {e}")
            errors.append(f"{valid_paths[0]}: {str(e)}")
    else:
        # Parallel editing for multiple images
        print(f"DEBUG - Parallel editing {len(valid_paths)} images...")
        with ThreadPoolExecutor(max_workers=len(valid_paths)) as executor:
            futures = {
                executor.submit(_edit_single_image, path, prompt): path
                for path in valid_paths
            }

            for future in as_completed(futures):
                original_path = futures[future]
                try:
                    edited_path = future.result()
                    edited_paths.append(edited_path)
                    print(f"DEBUG - Edited: {original_path} -> {edited_path}")
                except Exception as e:
                    print(f"DEBUG - Edit EXCEPTION for {original_path}: {type(e).__name__}: {e}")
                    errors.append(f"{original_path}: {str(e)}")

    et = time.time()
    print(f"DEBUG - Image editing completed in {et - st:.2f}s")

    if not edited_paths:
        print(f"DEBUG - All edits failed: {errors}")
        return f"Failed to edit images: {'; '.join(errors)}"

    print(f"DEBUG - Successfully edited {len(edited_paths)} image(s)")
    for path in edited_paths:
        print(f"DEBUG - Edited image path: {path}")

    result = f"Successfully edited {len(edited_paths)} image(s):\n"
    for path in edited_paths:
        result += f"- {path}\n"

    if errors:
        result += f"\nSome edits failed: {'; '.join(errors)}"

    print(f"DEBUG - Image edit result:\n{result}")
    return result
