"""
Download and cache the SDXL Lightning model weights.
"""

import os

import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0")
LIGHTNING_REPO = os.getenv("LIGHTNING_REPO", "ByteDance/SDXL-Lightning")
LIGHTNING_CKPT = os.getenv("LIGHTNING_CKPT", "sdxl_lightning_4step_unet.safetensors")


def download_models():
    """Download SDXL base and Lightning UNet weights."""
    print("Downloading SDXL Lightning components…")

    cache_dir = os.environ.get("HF_HOME", "/workspace/.cache/huggingface")
    os.makedirs(cache_dir, exist_ok=True)

    ckpt_path = hf_hub_download(LIGHTNING_REPO, LIGHTNING_CKPT)

    unet = UNet2DConditionModel.from_config(BASE_MODEL_ID, subfolder="unet")
    state_dict = load_file(ckpt_path, device="cpu")
    unet.load_state_dict(state_dict)

    pipe = StableDiffusionXLPipeline.from_pretrained(
        BASE_MODEL_ID,
        unet=unet,
        torch_dtype=torch.float16,
        variant="fp16",
    )

    _ = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    print("✅ SDXL Lightning cached successfully!")


if __name__ == "__main__":
    download_models()
