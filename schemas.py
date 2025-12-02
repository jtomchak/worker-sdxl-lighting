from dataclasses import dataclass
from typing import Optional


def _snap_dimension(value: int, minimum: int = 512, maximum: int = 1536) -> int:
    """Clamp and snap dimensions to multiples of 8 within a safe range."""
    clamped = max(minimum, min(maximum, value))
    return int(round(clamped / 8) * 8)


def _clamp_strength(value: float) -> float:
    """Clamp strength to valid range [0.0, 1.0]."""
    return max(0.0, min(1.0, value))


@dataclass
class LightningInput:
    prompt: str
    negative_prompt: Optional[str] = None

    # Optional input image URL for img2img
    image: Optional[str] = None
    strength: float = 0.8  # How much to transform the image (0.0 = no change, 1.0 = full change)

    height: int = 1024
    width: int = 1024

    num_inference_steps: int = 4
    guidance_scale: float = 0.0

    seed: Optional[int] = None
    num_images: int = 1

    @classmethod
    def from_event(cls, data: dict) -> "LightningInput":
        data = dict(data or {})
        if "height" in data:
            data["height"] = _snap_dimension(int(data["height"]))
        if "width" in data:
            data["width"] = _snap_dimension(int(data["width"]))
        if "strength" in data:
            data["strength"] = _clamp_strength(float(data["strength"]))
        return cls(**data)
