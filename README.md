![SDXL Turbo Worker Banner](https://cpjrphpz3t5wbwfe.public.blob.vercel-storage.com/worker-sdxl-turbo_banner.jpeg)

---

Run sdxl-turbo ([Stable Diffusion XL Turbo](https://huggingface.co/stabilityai/sdxl-turbo)) as a serverless endpoint for ultra-fast image generation. Supports both **Text-to-Image** and **Image-to-Image** modes.

---

[![RunPod](https://api.runpod.io/badge/runpod-workers/worker-sdxl-turbo)](https://www.runpod.io/console/hub/runpod-workers/worker-sdxl-turbo)

---

## Features

- **Text-to-Image**: Generate images from text prompts
- **Image-to-Image**: Transform existing images guided by text prompts
- **Ultra-fast**: 1-4 step generation
- **Batch generation**: Up to 4 images per request
- **Flexible dimensions**: 512-1536px with various aspect ratios

---

## Usage

The worker accepts the following input parameters:

| Parameter             | Type    | Default | Required | Description                                                            |
| :-------------------- | :------ | :------ | :------- | :--------------------------------------------------------------------- |
| `prompt`              | `str`   | `None`  | **Yes**  | The main text prompt describing the desired image                      |
| `negative_prompt`     | `str`   | `None`  | No       | Text prompt specifying concepts to exclude from the image              |
| `image`               | `str`   | `None`  | No       | URL of input image for Image-to-Image mode                             |
| `strength`            | `float` | `0.8`   | No       | Transformation intensity for img2img (0.0-1.0)                         |
| `height`              | `int`   | `1024`  | No       | Image height in pixels (512-1536, txt2img only)                        |
| `width`               | `int`   | `1024`  | No       | Image width in pixels (512-1536, txt2img only)                         |
| `num_inference_steps` | `int`   | `4`     | No       | Number of denoising steps (1-8)                                        |
| `guidance_scale`      | `float` | `0.0`   | No       | Guidance scale (0.0-20.0, recommended 0.0-2.0 for Turbo)               |
| `seed`                | `int`   | `None`  | No       | Random seed for reproducibility. If `None`, a random seed is used      |
| `num_images`          | `int`   | `1`     | No       | Number of images to generate per prompt (1-4)                          |

---

## Examples

### Text-to-Image

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

### Image-to-Image

```json
{
  "input": {
    "prompt": "Transform into a watercolor painting, artistic, vibrant colors",
    "image": "https://example.com/photo.jpg",
    "strength": 0.75,
    "num_inference_steps": 4,
    "seed": 42
  }
}
```

---

## Example Response

```json
{
  "delayTime": 2134,
  "executionTime": 1247,
  "id": "447f10b8-c745-4c3b-8fad-b1d4ebb7a65b-e1",
  "output": {
    "images": [
      {
        "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABAAAAAQACAIAAADwf7zU...",
        "seed": 1337
      }
    ],
    "generation_time": 0.85,
    "parameters": {
      "prompt": "a majestic steampunk dragon soaring through a cloudy sky...",
      "negative_prompt": "blurry, low quality, deformed, ugly",
      "width": 1024,
      "height": 1024,
      "num_inference_steps": 4,
      "guidance_scale": 0.0,
      "seed": 1337,
      "model": "stabilityai/sdxl-turbo"
    }
  },
  "status": "COMPLETED",
  "workerId": "462u6mrq9s28h6"
}
```

---

## API Documentation

For complete API reference, see [docs/API.md](docs/API.md) or the [OpenAPI specification](openapi.yaml).
