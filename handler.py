"""
SDXL Turbo Worker for RunPod Serverless
Ultra-fast image generation with Stable Diffusion XL Turbo
Supports both Text-to-Image and Image-to-Image modes
"""

import os
import base64
import io
import time
from typing import Optional, Dict, Any

import torch
import runpod
import requests
from runpod.serverless.utils import rp_upload, rp_cleanup
from runpod.serverless.utils.rp_validator import validate

from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
from PIL import Image

from schemas import INPUT_SCHEMA

# Timeout for downloading images (seconds)
IMAGE_DOWNLOAD_TIMEOUT = 30


class ModelHandler:
    def __init__(self):
        """Initialize the SDXL Turbo pipelines."""
        self.txt2img_pipe = None
        self.img2img_pipe = None
        self.load_models()

    def load_models(self):
        """Load the SDXL Turbo models for both txt2img and img2img."""
        print("🚀 Loading SDXL Turbo models...")

        try:
            # Load Text-to-Image pipeline
            self.txt2img_pipe = AutoPipelineForText2Image.from_pretrained(
                "stabilityai/sdxl-turbo",
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
                local_files_only=False,
            )

            # Load Image-to-Image pipeline from the same model (shares weights)
            self.img2img_pipe = AutoPipelineForImage2Image.from_pipe(
                self.txt2img_pipe
            )

            if torch.cuda.is_available():
                self.txt2img_pipe.to("cuda")
                self.img2img_pipe.to("cuda")
                print("✅ Models loaded successfully on GPU!")
            else:
                print("⚠️  GPU not available, running on CPU")

        except Exception as e:
            print(f"❌ Error loading models: {str(e)}")
            raise RuntimeError(f"Failed to load SDXL Turbo models: {str(e)}")

    def download_image(self, url: str) -> Image.Image:
        """Download an image from a URL."""
        print(f"📥 Downloading image from: {url[:50]}...")
        
        try:
            response = requests.get(url, timeout=IMAGE_DOWNLOAD_TIMEOUT)
            response.raise_for_status()
            
            image = Image.open(io.BytesIO(response.content))
            # Convert to RGB if necessary (handles RGBA, P mode, etc.)
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            print(f"✅ Image downloaded: {image.size[0]}x{image.size[1]}")
            return image
            
        except requests.exceptions.Timeout:
            raise RuntimeError(f"Image download timed out after {IMAGE_DOWNLOAD_TIMEOUT} seconds")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to download image: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Failed to process downloaded image: {str(e)}")

    def generate_text2img(self, job_input: Dict[str, Any]) -> Dict[str, Any]:
        """Generate image from text prompt."""
        
        prompt = job_input.get("prompt")
        negative_prompt = job_input.get("negative_prompt")
        height = 512
        width = 512
        num_inference_steps = job_input.get("num_inference_steps", 4)
        guidance_scale = job_input.get("guidance_scale", 0.0)
        num_images = job_input.get("num_images", 1)
        seed = job_input.get("seed")

        print(f"🎨 Text-to-Image: Generating {num_images} image(s)")
        print(f"📝 Prompt: '{prompt[:50]}...'")
        print(f"📐 Size: 512x512, Steps: {num_inference_steps}, Guidance: {guidance_scale}")

        # Set seed for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
            generator.manual_seed(seed)

        start_time = time.time()

        # Generate images
        result = self.txt2img_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images,
            generator=generator,
        )

        generation_time = time.time() - start_time
        print(f"⚡ Generated in {generation_time:.2f} seconds")

        # Process images
        images_data = []
        for i, image in enumerate(result.images):
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
            
            images_data.append({
                "image": f"data:image/png;base64,{image_b64}",
                "seed": seed + i if seed is not None else None
            })

        return {
            "images": images_data,
            "generation_time": generation_time,
            "parameters": {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "model": "stabilityai/sdxl-turbo",
            },
        }

    def generate_img2img(self, job_input: Dict[str, Any]) -> Dict[str, Any]:
        """Generate image from input image and text prompt."""
        
        prompt = job_input.get("prompt")
        negative_prompt = job_input.get("negative_prompt")
        image_url = job_input.get("image")
        strength = job_input.get("strength", 0.8)
        num_inference_steps = job_input.get("num_inference_steps", 4)
        guidance_scale = job_input.get("guidance_scale", 0.0)
        num_images = job_input.get("num_images", 1)
        seed = job_input.get("seed")

        print(f"🎨 Image-to-Image: Generating {num_images} image(s)")
        print(f"📝 Prompt: '{prompt[:50]}...'")
        print(f"💪 Strength: {strength}, Steps: {num_inference_steps}, Guidance: {guidance_scale}")

        # Download input image
        input_image = self.download_image(image_url)

        # Set seed for reproducibility
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
            generator.manual_seed(seed)

        start_time = time.time()

        # Generate images
        result = self.img2img_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=input_image,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images,
            generator=generator,
        )

        generation_time = time.time() - start_time
        print(f"⚡ Generated in {generation_time:.2f} seconds")

        # Process images
        images_data = []
        for i, image in enumerate(result.images):
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
            
            images_data.append({
                "image": f"data:image/png;base64,{image_b64}",
                "seed": seed + i if seed is not None else None
            })

        return {
            "images": images_data,
            "generation_time": generation_time,
            "parameters": {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "image": image_url,
                "strength": strength,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "model": "stabilityai/sdxl-turbo",
            },
        }

    def generate_image(self, job_input: Dict[str, Any]) -> Dict[str, Any]:
        """Generate image - routes to txt2img or img2img based on input."""
        
        image_url = job_input.get("image")
        
        try:
            if image_url:
                return self.generate_img2img(job_input)
            else:
                return self.generate_text2img(job_input)
        except Exception as e:
            print(f"❌ Error during generation: {str(e)}")
            raise RuntimeError(f"Image generation failed: {str(e)}")


# Initialize model handler
model_handler = ModelHandler()


def handler(job):
    """
    Handler function for RunPod serverless.
    """
    try:
        # Validate input
        job_input = job["input"]

        # Validate against schema
        validated_input = validate(job_input, INPUT_SCHEMA)
        if "errors" in validated_input:
            return {"error": f"Input validation failed: {validated_input['errors']}"}

        validated_data = validated_input["validated_input"]

        # Generate image
        result = model_handler.generate_image(validated_data)

        return result

    except Exception as e:
        print(f"❌ Handler error: {str(e)}")
        return {"error": str(e)}


if __name__ == "__main__":
    print("🎯 Starting SDXL Turbo Worker...")
    runpod.serverless.start({"handler": handler})
