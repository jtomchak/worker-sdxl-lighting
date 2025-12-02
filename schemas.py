from dataclasses import dataclass
from typing import Optional


def _snap_dimension(value: int, minimum: int = 512, maximum: int = 1536) -> int:
    """Clamp and snap dimensions to multiples of 8 within a safe range."""
    clamped = max(minimum, min(maximum, value))
    return int(round(clamped / 8) * 8)


@dataclass
class LightningInput:
    prompt: str
    negative_prompt: Optional[str] = None

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
        return cls(**data)
