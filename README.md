# FoxBurrow

FoxBurrow is a GPU inference server designed to keep your GPUs busy and your
clients happy. It pools multiple NVIDIA GPUs behind a unified REST + WebSocket
API, handling image generation, upscaling, background removal, and tagging. Jobs
are routed by capability and VRAM availability. Models are loaded on demand,
cached in GPU memory with LRU eviction, and can be pinned to survive pressure
spikes. If you're looking for a pretty web interface, this isn't it — FoxBurrow
is a workhorse, not a showroom.

## Quick Start

```
bin/foxburrow.sh
```

That's it. The script creates a Python virtual environment, installs dependencies,
and starts the server. On first run it auto-detects your GPUs via NVML, generates
a default `conf/foxburrow.ini` with a random API secret, and exits — asking you to
review the configuration and set `enabled=true` before it will actually start.

Run the script again after pulling new code. It only reinstalls packages when
`requirements.txt` changes.

### Requirements

- Python 3.11+
- NVIDIA GPU with CUDA support
- NVIDIA drivers installed (`nvidia-smi` should work)

See [docs/INSTALL.txt](docs/INSTALL.txt) for manual setup and model directory
layout.

## Configuration

FoxBurrow looks for `foxburrow.ini` in these locations (first match wins):

```
./foxburrow.ini          (project root)
./conf/foxburrow.ini     (standard location)
```

Each GPU gets its own `[GPU-<uuid>]` section with capabilities, onload, and
eviction settings. See [conf/foxburrow.ini.example](conf/foxburrow.ini.example)
for the full format.

### API Authentication

If `secret=` is set in `[server]`, all API requests must authenticate:

- **Header:** `Authorization: Bearer <secret>`
- **Query parameter:** `?apikey=<secret>`

Requests without valid credentials receive a `401 Unauthorized` response.
WebSocket connections are checked during the upgrade handshake.

## API Overview

All endpoints are served under the `/api` prefix.

### Synchronous

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/status` | GPU pool status, queue depth, VRAM usage |
| GET | `/api/model-list` | Available models |
| GET | `/api/lora-list` | Available LoRAs with metadata |
| POST | `/api/generate` | Generate an image (returns PNG) |
| POST | `/api/generate-hires` | Generate with hi-res fix |
| POST | `/api/upscale` | Upscale an image |
| POST | `/api/bgremove` | Remove background from an image |
| POST | `/api/tag` | Tag an image |

### Asynchronous (Job Queue)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/enqueue/{operation}` | Submit a job (returns `job_id`) |
| GET | `/api/job/{job_id}` | Poll job status |
| GET | `/api/job/{job_id}/result` | Retrieve completed result |
| WS | `/api/ws` | Real-time progress updates |

Operations: `generate`, `generate-hires`, `upscale`, `bgremove`, `tag`, `enhance`

### Latent Operations

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/generate-latents` | Generate latent tensor (no VAE decode) |
| POST | `/api/decode-latents` | VAE decode a latent tensor |
| POST | `/api/encode-latents` | VAE encode an image to latents |
| POST | `/api/hires-latents` | Hi-res fix in latent space |

## Models Directory

```
models/
  sdxl/              Checkpoints (.safetensors or diffusers directories)
  loras/             LoRA files (.safetensors)
  other/
    upscale/         Pixel upscale model
    bgremove/        Background removal model
    tagger/          Image tagging model
```

Not all directories are required. FoxBurrow runs with whatever models it finds
and logs what it discovered at startup.

## License

[MIT](LICENSE)
