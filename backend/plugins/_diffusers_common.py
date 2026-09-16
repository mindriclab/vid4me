"""Gemeinsame Helfer fuer Diffusers-basierte Plugins.

Kein LOADER-Attribut — dieses Modul wird von der Plugin-Discovery ignoriert.
torch/diffusers werden erst hier importiert, damit die API auch ohne
installierte ML-Abhaengigkeiten laeuft (Setup-Phase).
"""
from __future__ import annotations

import datetime
import json
import random
import re
from pathlib import Path
from typing import Optional

from backend.core import config
from backend.core.loader import GenerationRequest, LoraSpec, ProgressCallback


def get_torch():
    try:
        import torch
        return torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch ist nicht installiert. Bitte scripts/setup_backend.ps1 ausfuehren."
        ) from exc


def resolve_dtype(settings: dict):
    torch = get_torch()
    name = settings.get("inference", {}).get("dtype", "bfloat16")
    return {"bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32}.get(name, torch.bfloat16)


def model_source(entry: dict) -> str:
    """Lokalen Pfad bevorzugen, sonst Hugging-Face-Repo-ID."""
    local = config.resolve(entry["local_path"])
    if local.exists() and (local / "model_index.json").exists():
        return str(local)
    if not entry.get("repo_id"):
        raise RuntimeError(
            f"Modellgewichte nicht gefunden ({local}) und kein Download-Repo "
            "hinterlegt — Ordner wieder bereitstellen oder Modell neu importieren.")
    return entry["repo_id"]


def apply_offload(pipe, settings: dict) -> None:
    """Offload-Strategie aus settings.json anwenden (24GB-VRAM-tauglich)."""
    strategy = settings.get("inference", {}).get("offload", "model")
    if strategy == "model":
        pipe.enable_model_cpu_offload()
    elif strategy == "sequential":
        pipe.enable_sequential_cpu_offload()
    else:  # "none"
        pipe.to("cuda")
    if settings.get("inference", {}).get("vae_tiling", True) and hasattr(pipe, "vae"):
        try:
            pipe.vae.enable_tiling()
        except AttributeError:
            pass


def prepare_seed(req: GenerationRequest):
    torch = get_torch()
    seed = req.seed if req.seed is not None and req.seed >= 0 else random.randint(0, 2**32 - 1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return seed, generator


def make_step_callback(total_steps: int, progress_cb: Optional[ProgressCallback]):
    """callback_on_step_end fuer Diffusers-Pipelines."""
    def _cb(pipe, step_index, timestep, callback_kwargs):
        if progress_cb:
            progress_cb(step_index + 1, total_steps, f"Schritt {step_index + 1}/{total_steps}")
        return callback_kwargs
    return _cb


def output_path(req: GenerationRequest, extension: str) -> Path:
    folder = config.OUTPUTS_DIR / re.sub(r"[^\w\-. ]", "_", req.project or "default")
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return folder / f"{req.model_id}_{stamp}{extension}"


def write_sidecar(path: Path, req: GenerationRequest, seed: int, extra: dict | None = None) -> None:
    """Metadaten als JSON neben die Ausgabedatei schreiben (Reproduzierbarkeit)."""
    meta = {
        "model_id": req.model_id,
        "prompt": req.prompt,
        "negative_prompt": req.negative_prompt,
        "seed": seed,
        "steps": req.steps,
        "guidance_scale": req.guidance_scale,
        "width": req.width,
        "height": req.height,
        "lightning": req.lightning,
        "loras": [{"file": l.file, "strength": l.strength, "target": l.target}
                  for l in req.loras],
    }
    if extra:
        meta.update(extra)
    path.with_suffix(path.suffix + ".json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def lora_file_path(model_type: str, spec: LoraSpec) -> Path:
    path = config.LORA_DIRS[model_type] / spec.file
    if not path.exists():
        raise FileNotFoundError(f"LoRA-Datei nicht gefunden: {path}")
    return path


def sanitize_adapter_name(filename: str) -> str:
    return re.sub(r"[^\w]", "_", Path(filename).stem)
