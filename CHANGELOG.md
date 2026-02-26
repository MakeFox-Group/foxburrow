# Changelog

All notable changes to FoxBurrow will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-02-26

Initial standalone release, forked from the foxworksrv component of makefoxbot.

### Added
- Multi-GPU pooling with per-GPU capability routing and VRAM-aware scheduling
- LRU model cache with eviction, pinning (unevictable), and on-demand loading
- REST API for image generation, hi-res fix, upscaling, background removal, and tagging
- Async job queue with enqueue/poll endpoints and WebSocket progress streaming
- LoRA support with background fingerprint hashing and per-directory JSONL caching
- Regional prompting (attention-mode)
- First-start auto-configuration with NVML GPU detection
- Bearer token API authentication (header or query parameter)
- Safety gate requiring explicit `enabled=true` before server starts
- Single-script setup (`bin/foxburrow.sh`) — creates venv, installs deps, launches server
