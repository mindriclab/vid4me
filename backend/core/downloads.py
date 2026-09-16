"""Modell-Download-Manager.

Laedt Modellgewichte von Hugging Face direkt in den in der Registry
hinterlegten Ordner (models/...). Der eigentliche Download laeuft in einem
Subprozess (huggingface_hub.snapshot_download), damit er sauber abgebrochen
werden kann; der Fortschritt wird ueber die Ordnergroesse gemessen und per
WebSocket gebroadcastet ({"type": "download", ...}).
"""
from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from . import config
from .registry import registry


def _dir_size(path: Path) -> int:
    total = 0
    if path.exists():
        for p in path.rglob("*"):
            try:
                if p.is_file():
                    total += p.stat().st_size
            except OSError:
                continue
    return total


class DownloadManager:
    def __init__(self):
        self._states: dict[str, dict] = {}
        self._procs: dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()
        # Wird von der API gesetzt (WebSocket-Broadcast, threadsicher)
        self.broadcast: Callable[[dict], None] = lambda event: None

    # ---- Status ---------------------------------------------------------------

    def states(self) -> list[dict]:
        return list(self._states.values())

    def get(self, model_id: str) -> Optional[dict]:
        return self._states.get(model_id)

    def _emit(self, state: dict) -> None:
        self.broadcast({"type": "download", "download": state})

    # ---- Steuerung -------------------------------------------------------------

    def start(self, model_id: str) -> dict:
        entry = registry.get_model(model_id)
        if entry is None:
            raise KeyError(f"Unbekanntes Modell: {model_id}")
        if not entry.get("repo_id"):
            raise ValueError("Lokal importiertes Modell — kein Download-Repo hinterlegt")

        with self._lock:
            state = self._states.get(model_id)
            if state and state["status"] in ("starting", "running"):
                return state
            state = {
                "model_id": model_id,
                "repo_id": entry["repo_id"],
                "status": "starting",
                "progress": 0.0,
                "downloaded_gb": 0.0,
                "total_gb": None,
                "error": None,
            }
            self._states[model_id] = state

        threading.Thread(target=self._run, args=(model_id, entry),
                         daemon=True, name=f"download-{model_id}").start()
        self._emit(state)
        return state

    def cancel(self, model_id: str) -> bool:
        proc = self._procs.get(model_id)
        state = self._states.get(model_id)
        if proc is None or state is None or state["status"] not in ("starting", "running"):
            return False
        state["status"] = "cancelling"
        try:
            proc.terminate()
        except OSError:
            pass
        return True

    # ---- Worker -----------------------------------------------------------------

    def _total_bytes(self, repo_id: str, ignore: list[str]) -> Optional[int]:
        try:
            import fnmatch
            from huggingface_hub import HfApi
            info = HfApi().model_info(repo_id, files_metadata=True)
            total = 0
            for s in (info.siblings or []):
                if any(fnmatch.fnmatch(s.rfilename, pat) for pat in ignore):
                    continue
                total += s.size or 0
            return total or None
        except Exception:
            return None

    def _run(self, model_id: str, entry: dict) -> None:
        import os

        state = self._states[model_id]
        dest = config.resolve(entry["local_path"])
        dest.mkdir(parents=True, exist_ok=True)

        # Unnötige Repo-Dateien (z.B. Einzeldatei-Duplikate) überspringen
        ignore = list(entry.get("download_ignore", []))

        total = self._total_bytes(entry["repo_id"], ignore)
        if total:
            state["total_gb"] = round(total / 1024**3, 2)

        # hf_transfer (Rust-Downloader) nutzen, wenn installiert — deutlich
        # schneller bei grossen Dateien / schnellen Leitungen.
        env = dict(os.environ)
        try:
            import hf_transfer  # noqa: F401
            env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        except ImportError:
            pass

        code = (
            "from huggingface_hub import snapshot_download; "
            f"snapshot_download(repo_id={entry['repo_id']!r}, local_dir={str(dest)!r}, "
            f"max_workers=8, ignore_patterns={ignore!r})"
        )
        proc = subprocess.Popen(
            [sys.executable, "-c", code],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            text=True, env=env,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self._procs[model_id] = proc
        state["status"] = "running"
        self._emit(state)

        # Fortschritt ueber Ordnergroesse (inkl. Teil-Downloads im .cache)
        while proc.poll() is None:
            size = _dir_size(dest)
            state["downloaded_gb"] = round(size / 1024**3, 2)
            if total:
                state["progress"] = min(size / total, 0.999)
            self._emit(state)
            time.sleep(2)

        stderr_tail = ""
        if proc.stderr is not None:
            try:
                stderr_tail = (proc.stderr.read() or "")[-800:]
            except Exception:
                pass
        self._procs.pop(model_id, None)

        if state["status"] == "cancelling":
            state["status"] = "cancelled"
        elif proc.returncode == 0:
            state["status"] = "done"
            state["progress"] = 1.0
            state["downloaded_gb"] = round(_dir_size(dest) / 1024**3, 2)
        else:
            state["status"] = "error"
            state["error"] = stderr_tail.strip() or f"Download-Prozess endete mit Code {proc.returncode}"
        self._emit(state)


downloads = DownloadManager()
