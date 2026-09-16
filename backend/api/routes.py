"""REST-API fuer das Electron-Frontend."""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.core import config
from backend.core.downloads import downloads
from backend.core.jobs import jobs
from backend.core.loader import GenerationRequest, LoraSpec
from backend.core.registry import registry
from backend.core.vram import check_vram_for, query_vram

router = APIRouter(prefix="/api")

UPLOADS_DIR = config.ROOT / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

VIDEO_EXT = {".mp4", ".webm"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp"}


# ---- System ------------------------------------------------------------------

@router.get("/system")
def system_info():
    try:
        import torch  # noqa: F401
        torch_available = True
    except ImportError:
        torch_available = False
    try:
        import diffusers
        diffusers_version = diffusers.__version__
    except ImportError:
        diffusers_version = None
    return {
        "vram": query_vram(),
        "torch_available": torch_available,
        "diffusers_version": diffusers_version,
        "loaded_model": jobs.loaded_model_id,
        "plugins": registry.plugin_names(),
    }


@router.get("/system/vram-check/{model_id}")
def vram_check(model_id: str):
    entry = registry.get_model(model_id)
    if entry is None:
        raise HTTPException(404, f"Modell {model_id} nicht gefunden")
    return check_vram_for(entry.get("plugin", ""))


@router.post("/system/unload")
def unload_model():
    jobs.unload_model()
    return {"ok": True}


@router.get("/settings")
def get_settings():
    return config.load_settings()


class SettingsUpdate(BaseModel):
    settings: dict


@router.put("/settings")
def put_settings(body: SettingsUpdate):
    config.save_settings(body.settings)
    return {"ok": True}


# ---- Modelle -----------------------------------------------------------------

@router.get("/models")
def list_models(type: Optional[str] = None):
    return registry.models(type)


class ModelAdd(BaseModel):
    repo_id: str
    type: str            # video | image
    plugin: str          # z.B. wan22, chroma
    task: str            # t2v | i2v | t2i
    name: Optional[str] = None


@router.post("/models")
def add_model(body: ModelAdd):
    try:
        return registry.add_model(body.repo_id, body.type, body.plugin,
                                  body.task, body.name)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.delete("/models/{model_id}")
def delete_model(model_id: str):
    try:
        registry.delete_model(model_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"ok": True}


@router.delete("/models/{model_id}/files")
def delete_model_files(model_id: str):
    """Heruntergeladene Gewichte loeschen (Registry-Eintrag bleibt)."""
    dl = downloads.get(model_id)
    if dl and dl["status"] in ("starting", "running", "cancelling"):
        raise HTTPException(409, "Download laeuft noch — zuerst abbrechen.")
    if jobs.loaded_model_id == model_id:
        jobs.unload_model()
    try:
        freed_gb = registry.delete_model_files(model_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"ok": True, "freed_gb": freed_gb}


# ---- Lokale Modelle importieren -----------------------------------------------

@router.get("/models/local-scan")
def local_scan():
    """Unregistrierte Modell-Ordner in models/ finden."""
    return registry.scan_local_models()


class ModelImport(BaseModel):
    path: str
    type: str            # video | image
    plugin: str
    task: str            # t2v | i2v | t2i
    name: Optional[str] = None


@router.post("/models/import")
def import_model(body: ModelImport):
    try:
        return registry.import_local_model(body.path, body.type, body.plugin,
                                           body.task, body.name)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


class SingleFileImport(BaseModel):
    high_path: str
    low_path: Optional[str] = None   # leer = merged Checkpoint (ein Transformer)
    task: str                        # t2v | i2v
    name: Optional[str] = None


@router.post("/models/import-single")
def import_single_file(body: SingleFileImport):
    try:
        return registry.import_single_file(body.high_path, body.low_path,
                                           body.task, body.name)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


# ---- Modell-Downloads ------------------------------------------------------------

@router.get("/downloads")
def list_downloads():
    return downloads.states()


@router.post("/models/{model_id}/download")
def start_download(model_id: str):
    try:
        return downloads.start(model_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post("/models/{model_id}/download/cancel")
def cancel_download(model_id: str):
    if not downloads.cancel(model_id):
        raise HTTPException(409, "Kein laufender Download fuer dieses Modell")
    return {"ok": True}


@router.get("/hub/search")
def hub_search(q: str, limit: int = 15):
    """Modell-Suche auf dem Hugging Face Hub (benoetigt Internet)."""
    try:
        from huggingface_hub import HfApi
        results = HfApi().list_models(search=q, limit=limit, sort="downloads")
        return [{
            "repo_id": m.id,
            "downloads": getattr(m, "downloads", None),
            "likes": getattr(m, "likes", None),
            "pipeline_tag": getattr(m, "pipeline_tag", None),
        } for m in results]
    except Exception as exc:  # Netzwerkfehler etc. sauber melden
        raise HTTPException(502, f"Hub-Suche fehlgeschlagen: {exc}") from exc


# ---- LoRAs ---------------------------------------------------------------------

@router.get("/loras")
def list_loras():
    return registry.scan_loras()


class LoraMetaUpdate(BaseModel):
    name: Optional[str] = None
    strength_default: Optional[float] = None
    target: Optional[str] = None
    enabled_by_default: Optional[bool] = None


@router.patch("/loras/{model_type}/{filename}")
def update_lora(model_type: str, filename: str, body: LoraMetaUpdate):
    try:
        return registry.update_lora_meta(
            model_type, filename, body.model_dump(exclude_none=True))
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.delete("/loras/{model_type}/{filename}")
def delete_lora(model_type: str, filename: str):
    """LoRA-Datei und Registry-Eintrag loeschen."""
    if model_type not in config.LORA_DIRS:
        raise HTTPException(400, "model_type muss video oder image sein")
    try:
        registry.delete_lora(model_type, filename)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    return registry.scan_loras()


@router.post("/loras/{model_type}/upload")
async def upload_lora(model_type: str, file: UploadFile = File(...)):
    if model_type not in config.LORA_DIRS:
        raise HTTPException(400, "model_type muss video oder image sein")
    if not (file.filename or "").endswith(".safetensors"):
        raise HTTPException(400, "Nur .safetensors-Dateien werden akzeptiert")
    dest = config.LORA_DIRS[model_type] / Path(file.filename).name
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return registry.scan_loras()


# ---- LoRA-Presets ----------------------------------------------------------------

class PresetBody(BaseModel):
    name: str
    stack: list[dict] = Field(default_factory=list)  # [{file, strength, target}]


@router.get("/lora-presets/{model_type}")
def get_presets(model_type: str):
    return registry.presets(model_type)


@router.put("/lora-presets/{model_type}")
def save_preset(model_type: str, body: PresetBody):
    registry.save_preset(model_type, body.name, body.stack)
    return registry.presets(model_type)


@router.delete("/lora-presets/{model_type}/{name}")
def delete_preset(model_type: str, name: str):
    registry.delete_preset(model_type, name)
    return registry.presets(model_type)


# ---- Uploads (Startbild / Audio) ---------------------------------------------------

@router.post("/upload")
async def upload_input(file: UploadFile = File(...)):
    suffix = Path(file.filename or "upload.bin").suffix.lower()
    if suffix not in IMAGE_EXT | {".wav", ".mp3"}:
        raise HTTPException(400, f"Dateityp {suffix} nicht unterstuetzt")
    dest = UPLOADS_DIR / f"{uuid.uuid4().hex[:8]}{suffix}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"path": str(dest)}


# ---- Generierung -------------------------------------------------------------------

class LoraIn(BaseModel):
    file: str
    strength: float = 1.0
    target: str = "both"


class GenerateBody(BaseModel):
    model_id: str
    prompt: str
    negative_prompt: str = ""
    seed: int = -1
    steps: int = 40
    guidance_scale: float = 4.0
    width: int = 1024
    height: int = 1024
    num_frames: int = 81
    fps: int = 16
    image_path: Optional[str] = None
    audio_path: Optional[str] = None
    lightning: bool = False
    loras: list[LoraIn] = Field(default_factory=list)
    project: str = "default"


@router.post("/generate")
def generate(body: GenerateBody):
    model = next((m for m in registry.models() if m["id"] == body.model_id), None)
    if model is None:
        raise HTTPException(404, f"Modell {body.model_id} nicht gefunden")
    # Ohne diesen Check wuerde from_pretrained still das komplette Repo
    # (bis zu ~118 GB) von Hugging Face nachladen.
    if not model.get("downloaded"):
        raise HTTPException(
            409, f"Modell „{model.get('name', body.model_id)}“ ist nicht heruntergeladen. "
                 "Bitte zuerst ueber „⬇ Modelle“ laden oder ein anderes Modell waehlen.")
    if model.get("format") == "single_file" and not model.get("components_ready"):
        raise HTTPException(
            409, "Basis-Komponenten (VAE/Text-Encoder) fehlen. "
                 "Bitte zuerst „Wan 2.2 Basis-Komponenten“ ueber „⬇ Modelle“ laden.")
    req = GenerationRequest(
        **{**body.model_dump(exclude={"loras"}),
           "loras": [LoraSpec(**l.model_dump()) for l in body.loras]})
    job = jobs.submit(req)
    return job.public()


# ---- Jobs ---------------------------------------------------------------------------

@router.get("/jobs")
def list_jobs():
    return jobs.list()


@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job nicht gefunden")
    return job.public()


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    if not jobs.cancel(job_id):
        raise HTTPException(409, "Job kann nicht abgebrochen werden")
    return {"ok": True}


# ---- Outputs ---------------------------------------------------------------------------

HIDDEN_FILE = config.OUTPUTS_DIR / ".hidden.json"


def _load_hidden() -> set[str]:
    import json
    try:
        return set(json.loads(HIDDEN_FILE.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return set()


def _save_hidden(hidden: set[str]) -> None:
    import json
    HIDDEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    HIDDEN_FILE.write_text(json.dumps(sorted(hidden), indent=2), encoding="utf-8")


@router.get("/outputs")
def list_outputs(project: Optional[str] = None):
    base = config.OUTPUTS_DIR
    hidden = _load_hidden()
    folders = [base / project] if project else [p for p in base.iterdir() if p.is_dir()]
    files = []
    for folder in folders:
        if not folder.exists():
            continue
        for p in folder.iterdir():
            if p.suffix.lower() in VIDEO_EXT | IMAGE_EXT and str(p) not in hidden:
                files.append({
                    "path": str(p),
                    "name": p.name,
                    "project": folder.name,
                    "kind": "video" if p.suffix.lower() in VIDEO_EXT else "image",
                    "mtime": p.stat().st_mtime,
                })
    files.sort(key=lambda f: f["mtime"], reverse=True)
    return files


class OutputHide(BaseModel):
    path: str


@router.post("/outputs/hide")
def hide_output(body: OutputHide):
    """Ergebnis aus der Galerie ausblenden (Datei bleibt in outputs/)."""
    p = Path(body.path).resolve()
    if config.OUTPUTS_DIR.resolve() not in p.parents:
        raise HTTPException(403, "Pfad liegt nicht in outputs/")
    hidden = _load_hidden()
    hidden.add(body.path)  # exakt der String aus der Outputs-Liste
    _save_hidden(hidden)
    return {"ok": True}


@router.get("/file")
def serve_file(path: str):
    p = Path(path).resolve()
    # Nur Dateien innerhalb des Projekts ausliefern
    if config.ROOT.resolve() not in p.parents:
        raise HTTPException(403, "Pfad ausserhalb des Projekts")
    if not p.is_file():
        raise HTTPException(404, "Datei nicht gefunden")
    return FileResponse(p)
