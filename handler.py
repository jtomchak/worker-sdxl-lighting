"""
SDXL Lightning Worker for RunPod Serverless
Turbo-style API backed by ByteDance/SDXL-Lightning.
"""

import base64
import io
import os
import random
import time
from typing import Any, Dict, List, Optional

import requests
import runpod
import torch
from diffusers import (
    EulerDiscreteScheduler,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file

from schemas import LightningInput

# -------------------------------------------------------------------
# 1. Global model init
# -------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0")
LIGHTNING_REPO = os.getenv("LIGHTNING_REPO", "ByteDance/SDXL-Lightning")
LIGHTNING_CKPT = os.getenv("LIGHTNING_CKPT", "sdxl_lightning_4step_unet.safetensors")


def _infer_steps_from_ckpt(name: str) -> int:
    for n in (1, 2, 4, 8):
        if f"{n}step" in name:
            return n
    return 4


LIGHTNING_STEPS = _infer_steps_from_ckpt(LIGHTNING_CKPT)
print(f"[startup] Loading SDXL-Lightning ({LIGHTNING_CKPT}, {LIGHTNING_STEPS} steps) on {DEVICE}…")


def load_pipelines() -> tuple[StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline]:
    """Load both text2img and img2img pipelines with shared components."""
    ckpt_path = hf_hub_download(LIGHTNING_REPO, LIGHTNING_CKPT)

    unet = UNet2DConditionModel.from_config(
        BASE_MODEL_ID,
        subfolder="unet",
    ).to(DEVICE, DTYPE)

    state_dict = load_file(ckpt_path, device=DEVICE)
    unet.load_state_dict(state_dict)

    # Load text2img pipeline
    txt2img_pipe = StableDiffusionXLPipeline.from_pretrained(
        BASE_MODEL_ID,
        unet=unet,
        torch_dtype=DTYPE,
        variant="fp16" if DTYPE == torch.float16 else None,
    ).to(DEVICE)

    txt2img_pipe.scheduler = EulerDiscreteScheduler.from_config(
        txt2img_pipe.scheduler.config, timestep_spacing="trailing"
    )

    txt2img_pipe.enable_vae_slicing()
    try:
        txt2img_pipe.enable_xformers_memory_efficient_attention()
    except ModuleNotFoundError:
        print("[startup] xformers not available, using default attention")

    # Create img2img pipeline sharing components with txt2img
    img2img_pipe = StableDiffusionXLImg2ImgPipeline(
        vae=txt2img_pipe.vae,
        text_encoder=txt2img_pipe.text_encoder,
        text_encoder_2=txt2img_pipe.text_encoder_2,
        tokenizer=txt2img_pipe.tokenizer,
        tokenizer_2=txt2img_pipe.tokenizer_2,
        unet=txt2img_pipe.unet,
        scheduler=txt2img_pipe.scheduler,
    )

    return txt2img_pipe, img2img_pipe


txt2img_pipe, img2img_pipe = load_pipelines()


# -------------------------------------------------------------------
# 2. Utility functions
# -------------------------------------------------------------------

def _image_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def download_image(url: str, timeout: int = 30) -> Image.Image:
    """Download an image from a URL and return as PIL Image."""
    response = requests.get(url, timeout=timeout, stream=True)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")


# -------------------------------------------------------------------
# 3. RunPod handler
# -------------------------------------------------------------------

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    job_input = event.get("input") or {}
    li = LightningInput.from_event(job_input)

    if not li.prompt:
        raise ValueError("`prompt` is required.")

    num_inference_steps = LIGHTNING_STEPS
    guidance_scale = max(0.0, float(li.guidance_scale))

    if li.seed is None:
        li.seed = random.randint(0, 2**32 - 1)

    generator = torch.Generator(device=DEVICE).manual_seed(li.seed)
    num_images = max(1, min(4, li.num_images))

    # Check if this is an img2img request
    input_image: Optional[Image.Image] = None
    if li.image:
        input_image = download_image(li.image)

    start_time = time.time()
    with torch.inference_mode():
        if input_image is not None:
            # img2img: modify the input image with the prompt
            out = img2img_pipe(
                prompt=li.prompt,
                negative_prompt=li.negative_prompt,
                image=input_image,
                strength=li.strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images,
                generator=generator,
            )
        else:
            # txt2img: generate from scratch
            out = txt2img_pipe(
                prompt=li.prompt,
                negative_prompt=li.negative_prompt,
                height=li.height,
                width=li.width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images,
                generator=generator,
            )
    generation_time = round(time.time() - start_time, 3)

    images: List[Image.Image] = out.images
    data_urls = [_image_to_data_url(img) for img in images]

    parameters: Dict[str, Any] = {
        "prompt": li.prompt,
        "negative_prompt": li.negative_prompt,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "seed": li.seed,
        "model": "ByteDance/SDXL-Lightning",
        "checkpoint": LIGHTNING_CKPT,
    }

    if input_image is not None:
        parameters["image"] = li.image
        parameters["strength"] = li.strength
    else:
        parameters["width"] = li.width
        parameters["height"] = li.height

    return {
        "images": [
            {
                "image": data_urls[i],
                "seed": li.seed,
            }
            for i in range(len(data_urls))
        ],
        "generation_time": generation_time,
        "parameters": parameters,
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
