"""
SDXL Lightning Worker for RunPod Serverless
Turbo-style API backed by ByteDance/SDXL-Lightning.
"""

import base64
import io
import os
import random
from typing import Any, Dict, List

import runpod
import torch
from diffusers import (
    EulerDiscreteScheduler,
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


def load_pipeline() -> StableDiffusionXLPipeline:
    ckpt_path = hf_hub_download(LIGHTNING_REPO, LIGHTNING_CKPT)

    unet = UNet2DConditionModel.from_config(
        BASE_MODEL_ID,
        subfolder="unet",
    ).to(DEVICE, DTYPE)

    state_dict = load_file(ckpt_path, device=DEVICE)
    unet.load_state_dict(state_dict)

    pipe = StableDiffusionXLPipeline.from_pretrained(
        BASE_MODEL_ID,
        unet=unet,
        torch_dtype=DTYPE,
        variant="fp16" if DTYPE == torch.float16 else None,
    ).to(DEVICE)

    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )

    pipe.enable_vae_slicing()
    if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
        pipe.enable_xformers_memory_efficient_attention()

    return pipe


pipe = load_pipeline()


# -------------------------------------------------------------------
# 2. Utility functions
# -------------------------------------------------------------------

def _image_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


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

    with torch.inference_mode():
        out = pipe(
            prompt=li.prompt,
            negative_prompt=li.negative_prompt,
            height=li.height,
            width=li.width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images,
            generator=generator,
        )

    images: List[Image.Image] = out.images
    data_urls = [_image_to_data_url(img) for img in images]

    return {
        "images": [
            {
                "image": data_urls[i],
                "seed": li.seed,
            }
            for i in range(len(data_urls))
        ],
        "generation_time": None,
        "parameters": {
            "prompt": li.prompt,
            "negative_prompt": li.negative_prompt,
            "width": li.width,
            "height": li.height,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "seed": li.seed,
            "model": "ByteDance/SDXL-Lightning",
            "checkpoint": LIGHTNING_CKPT,
        },
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
