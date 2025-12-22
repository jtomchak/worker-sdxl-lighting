# SDXL Turbo API Reference

A high-performance image generation API powered by **stabilityai/sdxl-turbo** running on RunPod Serverless.

## Overview

SDXL Turbo is a distilled version of Stable Diffusion XL that generates high-quality images in just **1-4 inference steps**, making it significantly faster than standard SDXL while maintaining excellent image quality.

### Features

| Feature | Description |
|---------|-------------|
| **Text-to-Image** | Generate images from text prompts |
| **Image-to-Image** | Transform existing images guided by text prompts |
| **Ultra-fast** | 1-4 step generation |
| **Batch generation** | Up to 4 images per request |
| **Flexible dimensions** | 512-1536px with various aspect ratios |

---

## Authentication

All requests require a RunPod API key in the Authorization header:

```
Authorization: Bearer YOUR_RUNPOD_API_KEY
```

---

## Base URL

```
https://api.runpod.ai/v2/{endpoint_id}
```

Replace `{endpoint_id}` with your deployed endpoint ID.

---

## Endpoints

### POST `/runsync` - Synchronous Generation

Submit an image generation request and wait for the result (up to 90 seconds).

### POST `/run` - Asynchronous Generation

Submit a request and receive a job ID immediately. Poll `/status/{job_id}` for results.

### GET `/status/{job_id}` - Check Job Status

Retrieve the status and results of a submitted job.

### POST `/cancel/{job_id}` - Cancel Job

Cancel a queued or in-progress job.

---

## Request Body

All generation requests use the same input schema:

```json
{
  "input": {
    "prompt": "string (required)",
    "negative_prompt": "string (optional)",
    "image": "string URL (optional)",
    "strength": 0.8,
    "width": 1024,
    "height": 1024,
    "guidance_scale": 0.0,
    "seed": 42,
    "num_images": 1
  }
}
```

---

## Input Parameters

### `prompt` (required)
**Type:** `string`  
**Max Length:** 2000 characters

The text description of the image to generate.

**Tips for better results:**
- Be specific and descriptive
- Include style keywords (photorealistic, anime, oil painting, etc.)
- Mention lighting, composition, and quality modifiers

```json
"prompt": "A majestic lion in a savanna at sunset, photorealistic, 8k, detailed fur"
```

---

### `negative_prompt` (optional)
**Type:** `string | null`  
**Default:** `null`  
**Max Length:** 2000 characters

Text describing what to avoid in the generated image.

**Common negative prompts:**
```json
"negative_prompt": "blurry, low quality, distorted, deformed, ugly, bad anatomy, watermark"
```

---

### `image` (optional)
**Type:** `string (URL) | null`  
**Default:** `null`

URL of an input image for **Image-to-Image mode**.

When provided, the API transforms the input image using the prompt as guidance.

**Requirements:**
- Must be a publicly accessible URL
- Supported formats: JPEG, PNG, WebP
- Timeout: 30 seconds for download

```json
"image": "https://example.com/input-image.jpg"
```

---

### `strength` (optional)
**Type:** `float`  
**Default:** `0.8`  
**Range:** `0.0 - 1.0`

**Image-to-Image only.** Controls transformation intensity.

| Value | Effect |
|-------|--------|
| `0.0` | No change (returns original) |
| `0.3-0.5` | Light transformation, preserves most details |
| `0.6-0.8` | Moderate transformation (recommended) |
| `0.9-1.0` | Heavy transformation, may lose original structure |

---

### `width` (optional)
**Type:** `integer`  
**Default:** `1024`  
**Range:** `512 - 1536`

**Text-to-Image only.** Width in pixels. Values are snapped to multiples of 8.

**Recommended dimensions:**
| Aspect Ratio | Dimensions |
|--------------|------------|
| Square (1:1) | 1024 × 1024 |
| Landscape (16:9) | 1344 × 768 |
| Portrait (9:16) | 768 × 1344 |
| Wide (21:9) | 1536 × 640 |

---

### `height` (optional)
**Type:** `integer`  
**Default:** `1024`  
**Range:** `512 - 1536`

**Text-to-Image only.** Height in pixels. Values are snapped to multiples of 8.

---

### `guidance_scale` (optional)
**Type:** `float`  
**Default:** `0.0`  
**Range:** `0.0 - 20.0`

Controls how closely the generation follows the prompt.

**For SDXL Turbo:**
- `0.0` - Recommended for best quality (default)
- `1.0-2.0` - Slightly more prompt adherence

> ⚠️ **Note:** SDXL Turbo is trained for low/zero guidance. Higher values may reduce quality.

---

### `num_inference_steps` (optional)
**Type:** `integer`  
**Default:** `4`  
**Range:** `1 - 8`

Number of denoising steps. SDXL Turbo is optimized for 1-4 steps.

| Steps | Quality | Speed |
|-------|---------|-------|
| `1` | Good | Fastest |
| `2-4` | Best | Fast |
| `5-8` | Diminishing returns | Slower |

---

### `seed` (optional)
**Type:** `integer | null`  
**Default:** `null` (random)  
**Range:** `0 - 4294967295`

Random seed for reproducible generation.

- Same seed + same parameters = same image
- If not provided, a random seed is generated
- The used seed is returned in the response

---

### `num_images` (optional)
**Type:** `integer`  
**Default:** `1`  
**Range:** `1 - 4`

Number of images to generate per request.

---

## Response Format

### Successful Response

```json
{
  "id": "abc123-job-id",
  "status": "COMPLETED",
  "output": {
    "images": [
      {
        "image": "data:image/png;base64,iVBORw0KGgo...",
        "seed": 42
      }
    ],
    "generation_time": 1.234,
    "parameters": {
      "prompt": "A majestic lion in a savanna at sunset",
      "negative_prompt": null,
      "num_inference_steps": 4,
      "guidance_scale": 0.0,
      "seed": 42,
      "model": "stabilityai/sdxl-turbo",
      "width": 1024,
      "height": 1024
    }
  }
}
```

### Image-to-Image Response

For img2img requests, the parameters object includes `image` and `strength` instead of `width`/`height`:

```json
{
  "parameters": {
    "prompt": "...",
    "image": "https://example.com/input.jpg",
    "strength": 0.75,
    "...": "..."
  }
}
```

### Job Status Values

| Status | Description |
|--------|-------------|
| `IN_QUEUE` | Job waiting to be processed |
| `IN_PROGRESS` | Job currently being processed |
| `COMPLETED` | Job finished successfully |
| `FAILED` | Job failed (check `error` field) |
| `CANCELLED` | Job was cancelled |

---

## Examples

### cURL Examples

#### Basic Text-to-Image

```bash
curl -X POST "https://api.runpod.ai/v2/{endpoint_id}/runsync" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "A majestic lion in a savanna at sunset, photorealistic, 8k"
    }
  }'
```

#### Advanced Text-to-Image

```bash
curl -X POST "https://api.runpod.ai/v2/{endpoint_id}/runsync" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "A cyberpunk cityscape at night, neon lights reflecting on wet streets, highly detailed, cinematic lighting",
      "negative_prompt": "blurry, low quality, distorted, deformed",
      "width": 1344,
      "height": 768,
      "guidance_scale": 1.5,
      "seed": 42,
      "num_images": 2
    }
  }'
```

#### Image-to-Image

```bash
curl -X POST "https://api.runpod.ai/v2/{endpoint_id}/runsync" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "Transform into a watercolor painting, artistic, vibrant colors",
      "image": "https://example.com/photo.jpg",
      "strength": 0.75
    }
  }'
```

#### Asynchronous Request

```bash
# Submit job
curl -X POST "https://api.runpod.ai/v2/{endpoint_id}/run" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "A fantasy landscape with floating islands",
      "num_images": 4
    }
  }'

# Response: {"id": "abc123", "status": "IN_QUEUE"}

# Poll for results
curl "https://api.runpod.ai/v2/{endpoint_id}/status/abc123" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### Python Example

```python
import requests
import base64
from io import BytesIO
from PIL import Image

ENDPOINT_ID = "your-endpoint-id"
API_KEY = "your-api-key"

def generate_image(prompt, **kwargs):
    response = requests.post(
        f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={"input": {"prompt": prompt, **kwargs}}
    )
    response.raise_for_status()
    return response.json()

def decode_image(data_url):
    """Convert data URL to PIL Image."""
    base64_data = data_url.split(",")[1]
    image_data = base64.b64decode(base64_data)
    return Image.open(BytesIO(image_data))

# Text-to-Image
result = generate_image(
    prompt="A serene mountain lake at dawn, mist rising from the water",
    width=1344,
    height=768,
    seed=42
)

for i, img_data in enumerate(result["output"]["images"]):
    image = decode_image(img_data["image"])
    image.save(f"output_{i}.png")
    print(f"Saved output_{i}.png (seed: {img_data['seed']})")

# Image-to-Image
result = generate_image(
    prompt="In the style of Studio Ghibli anime",
    image="https://example.com/photo.jpg",
    strength=0.7
)
```

### JavaScript/Node.js Example

```javascript
const fetch = require('node-fetch');
const fs = require('fs');

const ENDPOINT_ID = 'your-endpoint-id';
const API_KEY = 'your-api-key';

async function generateImage(input) {
  const response = await fetch(
    `https://api.runpod.ai/v2/${ENDPOINT_ID}/runsync`,
    {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${API_KEY}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ input })
    }
  );
  
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  
  return response.json();
}

function saveDataUrl(dataUrl, filename) {
  const base64Data = dataUrl.replace(/^data:image\/\w+;base64,/, '');
  const buffer = Buffer.from(base64Data, 'base64');
  fs.writeFileSync(filename, buffer);
}

// Usage
(async () => {
  const result = await generateImage({
    prompt: 'A futuristic robot in a garden of flowers',
    width: 1024,
    height: 1024
  });
  
  result.output.images.forEach((img, i) => {
    saveDataUrl(img.image, `output_${i}.png`);
    console.log(`Saved output_${i}.png (seed: ${img.seed})`);
  });
  
  console.log(`Generation time: ${result.output.generation_time}s`);
})();
```

---

## Error Handling

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `"prompt" is required` | Missing prompt | Add a prompt to your request |
| `401 Unauthorized` | Invalid API key | Check your RunPod API key |
| `408 Timeout` | Generation took too long | Use `/run` + polling instead |
| Image download failed | Invalid image URL | Ensure URL is publicly accessible |

### Error Response Format

```json
{
  "id": "abc123",
  "status": "FAILED",
  "error": "Error message describing what went wrong"
}
```

---

## Rate Limits & Best Practices

1. **Use async for batches** - For multiple images or large dimensions, use `/run` with polling
2. **Reuse seeds** - Save seeds from good generations to reproduce similar results
3. **Start with defaults** - The default `guidance_scale: 0.0` works best for Turbo
4. **Optimize dimensions** - Use standard aspect ratios for best quality
5. **Queue management** - Monitor your RunPod queue to avoid timeouts

---

## Model Information

| Property | Value |
|----------|-------|
| Model | `stabilityai/sdxl-turbo` |
| Base | `stabilityai/stable-diffusion-xl-base-1.0` |
| Default Steps | 4 |
| Recommended Guidance | 0.0 - 2.0 |

---

## OpenAPI Specification

The complete OpenAPI 3.0 specification is available at [`openapi.yaml`](../openapi.yaml).

You can use it with:
- [Swagger UI](https://swagger.io/tools/swagger-ui/)
- [Postman](https://www.postman.com/)
- [Insomnia](https://insomnia.rest/)
- Any OpenAPI-compatible tool
