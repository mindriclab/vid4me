# Vid4me (archived)

**Local image and video generation desktop app. Electron UI, Python backend, model plugins, LoRA stacking. Fully offline, no cloud APIs.**

> **Archived, July 2026.** Paused once it became clear that ComfyUI would always be ahead of the model ecosystem: new models and fine-tunes appear in ComfyUI formats first, often as custom fp8 checkpoints that diffusers cannot load, and ComfyUI runs the same models faster on the same GPU. The goal of a cleaner, more intuitive UI continues in [ComfyDeck](https://github.com/mindriclab/comfydeck), a thin front-end on top of ComfyUI's API. Details in [STATUS.md](STATUS.md).
>
> German documentation: [README.de.md](README.de.md)

## What was built

- Electron + React (Vite) front-end, Python + FastAPI backend, started and stopped automatically by the app
- Plugin system for models: Wan 2.2 T2V / I2V (A14B), Chroma1-HD, LTX-2.3 (untested with real weights)
- Model download manager with resumable downloads, single-file checkpoint import (Wan), generic LoRA stacking
- Portable Windows build that sets up its own Python environment on first start

## Stack

Electron · React · Vite · Python 3.12 · FastAPI · diffusers · PyTorch (CUDA 12.8)

## Running it (for the curious)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_backend.ps1   # once: venv, torch, diffusers (~3 GB)
cd frontend && npm install
npm run dev
```

Models are large (Chroma ~26 GB, each Wan variant ~118 GB) and are downloaded inside the app.

## License

MIT · Rico Herrmann · [mindriclab.de](https://www.mindriclab.de)
