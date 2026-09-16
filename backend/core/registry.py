"""Modell-Registry mit Plugin-Discovery und LoRA-Auto-Erkennung.

- Modelle stehen in models/registry.json und referenzieren ein Plugin
  (backend/plugins/<name>.py). Neue Modell-Familien = neues Plugin-Modul,
  kein Core-Code noetig.
- LoRA-Dateien (.safetensors) in models/loras/{video,image} werden bei jedem
  Scan automatisch erkannt und mit Default-Metadaten in registry.json
  aufgenommen.
"""
from __future__ import annotations

import importlib
import json
import pkgutil
import threading
from pathlib import Path
from typing import Optional

from . import config
from .loader import ModelLoader

_lock = threading.Lock()


class Registry:
    def __init__(self):
        self._data: dict = {}
        self._plugins: dict[str, type[ModelLoader]] = {}
        self.reload()
        self.discover_plugins()

    # ---- registry.json -----------------------------------------------------

    def reload(self) -> None:
        with config.REGISTRY_FILE.open("r", encoding="utf-8") as f:
            self._data = json.load(f)

    def save(self) -> None:
        with _lock:
            tmp = config.REGISTRY_FILE.with_suffix(".json.tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
            tmp.replace(config.REGISTRY_FILE)

    # ---- Plugins -----------------------------------------------------------

    def discover_plugins(self) -> None:
        """Alle Module in backend/plugins importieren und deren Loader registrieren."""
        import backend.plugins as plugins_pkg

        self._plugins.clear()
        for mod_info in pkgutil.iter_modules(plugins_pkg.__path__):
            module = importlib.import_module(f"backend.plugins.{mod_info.name}")
            loader_cls = getattr(module, "LOADER", None)
            if loader_cls is not None and issubclass(loader_cls, ModelLoader):
                self._plugins[loader_cls.plugin_name] = loader_cls

    def plugin_names(self) -> list[str]:
        return sorted(self._plugins)

    # ---- Modelle -----------------------------------------------------------

    def models(self, model_type: Optional[str] = None) -> list[dict]:
        models = self._data.get("models", [])
        if model_type:
            models = [m for m in models if m.get("type") == model_type]
        # Verfuegbarkeit pruefen: lokale Gewichte vorhanden UND vollstaendig?
        result = []
        for m in models:
            m = dict(m)
            if m.get("format") == "single_file":
                files = m.get("single_files", {})
                paths = [config.resolve(p) for p in files.values()]
                m["downloaded"] = bool(paths) and all(p.exists() for p in paths)
                m["components_ready"] = self._components_ready()
            else:
                m["downloaded"] = self._is_complete(config.resolve(m["local_path"]))
            m["plugin_available"] = m.get("plugin") in self._plugins
            result.append(m)
        return result

    def _components_ready(self) -> bool:
        comp = self.get_model("wan22-components")
        return comp is not None and self._is_complete(config.resolve(comp["local_path"]))

    @staticmethod
    def _is_complete(local) -> bool:
        """Diffusers-Ordner gilt als vorhanden, wenn model_index.json existiert
        und keine unfertigen Download-Dateien (.incomplete) mehr da sind."""
        if not local.exists() or not (local / "model_index.json").exists():
            return False
        cache = local / ".cache"
        if cache.exists() and any(cache.rglob("*.incomplete")):
            return False
        return True

    def get_model(self, model_id: str) -> Optional[dict]:
        for m in self._data.get("models", []):
            if m["id"] == model_id:
                return m
        return None

    def add_model(self, repo_id: str, model_type: str, plugin: str,
                  task: str, name: Optional[str] = None) -> dict:
        """Eigenes Modell (Hugging-Face-Repo) in die Registry aufnehmen."""
        import re
        if model_type not in ("video", "image"):
            raise ValueError("type muss video oder image sein")
        if plugin not in self._plugins:
            raise ValueError(f"Unbekanntes Plugin: {plugin} "
                             f"(verfuegbar: {', '.join(self.plugin_names())})")
        slug = re.sub(r"[^\w\-]", "-", repo_id.lower()).strip("-")
        model_id = slug
        if self.get_model(model_id) is not None:
            raise ValueError(f"Modell {model_id} existiert bereits")
        entry = {
            "id": model_id,
            "name": name or repo_id,
            "type": model_type,
            "task": task,
            "plugin": plugin,
            "repo_id": repo_id,
            "local_path": f"models/{model_type}/{slug}",
            "capabilities": ["lora"],
            "custom": True,
        }
        self._data.setdefault("models", []).append(entry)
        self.save()
        return entry

    def delete_model(self, model_id: str) -> None:
        models = self._data.get("models", [])
        entry = self.get_model(model_id)
        if entry is None:
            raise KeyError(f"Modell nicht gefunden: {model_id}")
        models.remove(entry)
        self.save()

    def delete_model_files(self, model_id: str) -> float:
        """Heruntergeladene Gewichte eines Modells loeschen (Registry-Eintrag
        bleibt). Nur Pfade innerhalb von models/ werden geloescht. Gibt die
        freigegebenen GB zurueck."""
        import shutil

        entry = self.get_model(model_id)
        if entry is None:
            raise KeyError(f"Modell nicht gefunden: {model_id}")
        local = config.resolve(entry["local_path"]).resolve()
        models_dir = config.MODELS_DIR.resolve()
        if models_dir not in local.parents:
            raise ValueError(
                "Pfad liegt ausserhalb des models/-Ordners der App — "
                "extern eingebundene Modelle bitte manuell loeschen.")
        if not local.exists():
            return 0.0
        size = sum(p.stat().st_size for p in local.rglob("*") if p.is_file())
        shutil.rmtree(local)
        return round(size / 1024**3, 2)

    # ---- Lokale Modelle importieren -----------------------------------------

    def scan_local_models(self) -> list[dict]:
        """models/video und models/image nach unregistrierten Modellen
        durchsuchen: Diffusers-Ordner (model_index.json), Ordner mit
        .safetensors sowie lose .safetensors-Dateien (Einzeldatei-Checkpoints,
        ComfyUI-Format)."""
        registered = set()
        for m in self._data.get("models", []):
            registered.add(str(config.resolve(m["local_path"]).resolve()))
            for p in m.get("single_files", {}).values():
                registered.add(str(config.resolve(p).resolve()))

        candidates = []
        for model_type in ("video", "image"):
            base = config.MODELS_DIR / model_type
            if not base.exists():
                continue

            # Lose Einzeldateien direkt in models/<typ>/
            for f in sorted(base.glob("*.safetensors")):
                if str(f.resolve()) in registered:
                    continue
                candidates.append({
                    "path": str(f),
                    "name": f.stem,
                    "type": model_type,
                    "format": "single_file",
                    "size_gb": round(f.stat().st_size / 1024**3, 2),
                })

            def under_registered(p) -> bool:
                rp = str(p.resolve())
                return any(rp == reg or rp.startswith(reg + "\\") or rp.startswith(reg + "/")
                           for reg in registered)

            dirs = [d for d in base.iterdir() if d.is_dir() and not under_registered(d)]
            dirs += [c for d in dirs for c in d.iterdir()
                     if c.is_dir() and not under_registered(c)]
            for d in dirs:
                if (d / "model_index.json").exists():
                    size = sum(p.stat().st_size for p in d.rglob("*") if p.is_file())
                    candidates.append({
                        "path": str(d),
                        "name": d.name,
                        "type": model_type,
                        "format": "diffusers",
                        "size_gb": round(size / 1024**3, 2),
                    })
                else:
                    # Einzeldateien in Unterordnern einzeln anbieten
                    for f in sorted(d.glob("*.safetensors")):
                        if str(f.resolve()) in registered:
                            continue
                        candidates.append({
                            "path": str(f),
                            "name": f.stem,
                            "type": model_type,
                            "format": "single_file",
                            "size_gb": round(f.stat().st_size / 1024**3, 2),
                        })
        return candidates

    def import_single_file(self, high_path: str, low_path: Optional[str],
                           task: str, name: Optional[str] = None) -> dict:
        """Einzeldatei-Checkpoint (ComfyUI-Format, nur Wan-Video) einbinden.

        high_path: Checkpoint fuer den High-Noise-Experten ODER ein
                   kombiniertes/merged Checkpoint (dann low_path leer lassen —
                   die Pipeline laeuft dann mit nur einem Transformer).
        low_path:  optional der Low-Noise-Checkpoint (Wan 2.2 MoE-Paar).
        """
        import re

        high = config.resolve(high_path)
        if not high.is_file() or high.suffix != ".safetensors":
            raise ValueError(f".safetensors-Datei nicht gefunden: {high_path}")
        low = None
        if low_path:
            low = config.resolve(low_path)
            if not low.is_file() or low.suffix != ".safetensors":
                raise ValueError(f".safetensors-Datei nicht gefunden: {low_path}")
        if task not in ("t2v", "i2v"):
            raise ValueError("task muss t2v oder i2v sein")

        def rel(p):
            try:
                return str(p.resolve().relative_to(config.ROOT.resolve())).replace("\\", "/")
            except ValueError:
                return str(p.resolve())

        display = name or high.stem
        slug = re.sub(r"[^\w\-]", "-", f"single-{display}".lower()).strip("-")
        model_id = slug
        n = 2
        while self.get_model(model_id) is not None:
            model_id = f"{slug}-{n}"
            n += 1

        single_files = {"high": rel(high)}
        if low is not None:
            single_files["low"] = rel(low)

        entry = {
            "id": model_id,
            "name": display,
            "type": "video",
            "task": task,
            "plugin": "wan22",
            "repo_id": None,
            "format": "single_file",
            "single_files": single_files,
            "local_path": rel(high),
            "capabilities": ["lora"],
            "custom": True,
            "imported": True,
        }
        self._data.setdefault("models", []).append(entry)
        self.save()
        return entry

    def import_local_model(self, path: str, model_type: str, plugin: str,
                           task: str, name: Optional[str] = None) -> dict:
        """Einen lokal vorhandenen Modell-Ordner (Diffusers-Format) in die
        Registry aufnehmen. Der Ordner kann in models/ liegen oder ein
        beliebiger absoluter Pfad sein (wird dann in-place eingebunden)."""
        import re

        p = config.resolve(path)
        if not p.exists() or not p.is_dir():
            raise ValueError(f"Ordner nicht gefunden: {path}")
        if not (p / "model_index.json").exists():
            raise ValueError(
                "Kein Diffusers-Format (model_index.json fehlt). Einzeldatei-"
                "Checkpoints (ComfyUI-Format) werden noch nicht unterstuetzt — "
                "bitte die Diffusers-Version des Modells verwenden.")
        if model_type not in ("video", "image"):
            raise ValueError("type muss video oder image sein")
        if plugin not in self._plugins:
            raise ValueError(f"Unbekanntes Plugin: {plugin}")

        # Pfad relativ zum Root speichern, wenn moeglich (portabel)
        try:
            local_path = str(p.resolve().relative_to(config.ROOT.resolve())).replace("\\", "/")
        except ValueError:
            local_path = str(p.resolve())

        display = name or p.name
        slug = re.sub(r"[^\w\-]", "-", f"local-{display}".lower()).strip("-")
        model_id = slug
        n = 2
        while self.get_model(model_id) is not None:
            model_id = f"{slug}-{n}"
            n += 1

        entry = {
            "id": model_id,
            "name": display,
            "type": model_type,
            "task": task,
            "plugin": plugin,
            "repo_id": None,
            "local_path": local_path,
            "capabilities": ["lora"],
            "custom": True,
            "imported": True,
        }
        self._data.setdefault("models", []).append(entry)
        self.save()
        return entry

    def create_loader(self, model_id: str, settings: dict) -> ModelLoader:
        entry = self.get_model(model_id)
        if entry is None:
            raise KeyError(f"Unbekanntes Modell: {model_id}")
        plugin = self._plugins.get(entry.get("plugin", ""))
        if plugin is None:
            raise KeyError(f"Plugin nicht gefunden: {entry.get('plugin')}")
        return plugin(entry, settings)

    # ---- LoRAs -------------------------------------------------------------

    @staticmethod
    def _guess_lora_target(filename: str) -> str:
        """Wan-2.2-Paar-LoRAs am Dateinamen erkennen ("HighNoise", "low_noise",
        "_high_" usw.) und das Experten-Ziel vorbelegen."""
        import re
        if re.search(r"high[\s._-]?noise|[._-]high(?=[._-])", filename, re.I):
            return "high"
        if re.search(r"low[\s._-]?noise|[._-]low(?=[._-])", filename, re.I):
            return "low"
        return "both"

    def scan_loras(self) -> dict:
        """LoRA-Ordner scannen; neue Dateien mit Default-Metadaten aufnehmen,
        verschwundene Dateien als missing markieren (Metadaten bleiben erhalten)."""
        changed = False
        loras = self._data.setdefault("loras", {"video": {}, "image": {}})
        for model_type, folder in config.LORA_DIRS.items():
            known: dict = loras.setdefault(model_type, {})
            found = {p.name for p in folder.glob("*.safetensors")} if folder.exists() else set()
            for name in sorted(found):
                if name not in known:
                    known[name] = {
                        "name": Path(name).stem,
                        "strength_default": 1.0,
                        "model_type": model_type,
                        "target": self._guess_lora_target(name) if model_type == "video" else "both",
                        "enabled_by_default": False,
                    }
                    changed = True
                if known[name].get("missing"):
                    known[name].pop("missing", None)
                    changed = True
            for name, meta in known.items():
                if name not in found and not meta.get("missing"):
                    meta["missing"] = True
                    changed = True
        if changed:
            self.save()
        return loras

    def delete_lora(self, model_type: str, filename: str) -> None:
        """LoRA-Datei loeschen und aus der Registry entfernen."""
        loras = self._data.setdefault("loras", {}).setdefault(model_type, {})
        if filename not in loras:
            raise KeyError(f"LoRA nicht gefunden: {filename}")
        path = config.LORA_DIRS[model_type] / Path(filename).name
        if path.exists():
            path.unlink()
        del loras[filename]
        self.save()

    def update_lora_meta(self, model_type: str, filename: str, meta: dict) -> dict:
        loras = self._data.setdefault("loras", {}).setdefault(model_type, {})
        if filename not in loras:
            raise KeyError(f"LoRA nicht gefunden: {filename}")
        allowed = {"name", "strength_default", "target", "enabled_by_default"}
        loras[filename].update({k: v for k, v in meta.items() if k in allowed})
        self.save()
        return loras[filename]

    # ---- LoRA-Presets --------------------------------------------------------

    def presets(self, model_type: str) -> dict:
        return self._data.setdefault("lora_presets", {}).setdefault(model_type, {})

    def save_preset(self, model_type: str, name: str, stack: list[dict]) -> None:
        self.presets(model_type)[name] = stack
        self.save()

    def delete_preset(self, model_type: str, name: str) -> None:
        presets = self.presets(model_type)
        if name in presets:
            del presets[name]
            self.save()


registry = Registry()
