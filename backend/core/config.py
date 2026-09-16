"""Zentrale Pfad- und Settings-Verwaltung fuer Vid4me."""
from __future__ import annotations

import json
import threading
from pathlib import Path

# Daten-Root: In der gepackten App zeigt VID4ME_ROOT auf den Datenordner
# (z.B. %USERPROFILE%\Vid4me); im Dev-Modus ist es der Projektordner
# (zwei Ebenen ueber dieser Datei).
import os

ROOT = Path(os.environ.get("VID4ME_ROOT") or Path(__file__).resolve().parents[2])

CONFIG_FILE = ROOT / "config" / "settings.json"
REGISTRY_FILE = ROOT / "models" / "registry.json"
OUTPUTS_DIR = ROOT / "outputs"
MODELS_DIR = ROOT / "models"
LORA_DIRS = {
    "video": ROOT / "models" / "loras" / "video",
    "image": ROOT / "models" / "loras" / "image",
}

_lock = threading.Lock()


def load_settings() -> dict:
    with CONFIG_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_settings(settings: dict) -> None:
    with _lock:
        tmp = CONFIG_FILE.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
        tmp.replace(CONFIG_FILE)


def resolve(path_str: str) -> Path:
    """Relative Pfade aus settings/registry relativ zum Projekt-Root aufloesen."""
    p = Path(path_str)
    return p if p.is_absolute() else ROOT / p
