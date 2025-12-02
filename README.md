![SDXL Lightning Worker Banner](https://cpjrphpz3t5wbwfe.public.blob.vercel-storage.com/worker-sdxl-turbo_banner.jpeg)

---

Run SDXL Lightning ([ByteDance/SDXL-Lightning](https://huggingface.co/ByteDance/SDXL-Lightning)) as a serverless endpoint with the same API surface as the SDXL Turbo worker.

---

[![RunPod](https://api.runpod.io/badge/runpod-workers/worker-sdxl-turbo)](https://www.runpod.io/console/hub/runpod-workers/worker-sdxl-turbo)

---

## Usage

The worker accepts the following input parameters:

| Parameter             | Type    | Default | Required | Description                                                       |
| :-------------------- | :------ | :------ | :------- | :---------------------------------------------------------------- |
| `prompt`              | `str`   | `None`  | **Yes**  | The main text prompt describing the desired image                 |
| `negative_prompt`     | `str`   | `None`  | No       | Text prompt specifying concepts to exclude from the image         |
| `height`              | `int`   | `1024`  | No       | The height of the generated image in pixels (snapped to multiples of 8 between 512-1536) |
| `width`               | `int`   | `1024`  | No       | The width of the generated image in pixels (snapped to multiples of 8 between 512-1536)  |
| `num_inference_steps` | `int`   | `4`     | No       | Number of denoising steps (clamped to the Lightning checkpoint)   |
| `guidance_scale`      | `float` | `0.0`   | No       | Guidance scale (Lightning prefers 0.0)                            |
| `seed`                | `int`   | `None`  | No       | Random seed for reproducibility. If `None`, a random seed is used |
| `num_images`          | `int`   | `1`     | No       | Number of images to generate per prompt (Constraint: must be 1-4) |

### Example Request

```json
{
  "input": {
    "prompt": "a majestic steampunk dragon soaring through a cloudy sky, intricate clockwork details, golden hour lighting, highly detailed",
    "negative_prompt": "blurry, low quality, deformed, ugly",
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 4,
    "guidance_scale": 0.0,
    "seed": 1337,
    "num_images": 1
  }
}
```

### Example Response

```json
{
  "output": {
    "images": [
      {
        "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABAAAAAQACAIAAADwf7zU...",
        "seed": 1337
      }
    ],
    "generation_time": null,
    "parameters": {
      "prompt": "a majestic steampunk dragon soaring through a cloudy sky, intricate clockwork details, golden hour lighting, highly detailed",
      "negative_prompt": "blurry, low quality, deformed, ugly",
      "width": 1024,
      "height": 1024,
      "num_inference_steps": 4,
      "guidance_scale": 0.0,
      "seed": 1337,
      "model": "ByteDance/SDXL-Lightning",
      "checkpoint": "sdxl_lightning_4step_unet.safetensors"
    }
  },
  "status": "COMPLETED"
}
```
