"""Job-Queue fuer Generierungen.

GPU-Arbeit laeuft seriell in einem Worker-Thread. Fortschritt wird ueber
einen Broadcast-Hook (WebSocket) an das Frontend gemeldet. Es bleibt immer
nur ein Modell geladen; bei Modellwechsel wird das alte entladen.
"""
from __future__ import annotations

import queue
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

from .loader import GenerationRequest, ModelLoader
from .registry import registry


@dataclass
class Job:
    id: str
    request: GenerationRequest
    status: str = "queued"          # queued | loading | running | done | error | cancelled
    progress: float = 0.0           # 0..1
    step: int = 0
    total_steps: int = 0
    message: str = ""
    output_path: Optional[str] = None
    seed: Optional[int] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None

    def public(self) -> dict:
        d = asdict(self)
        d["request"] = {k: v for k, v in asdict(self.request).items()
                        if k not in ("extra",)}
        return d


class JobManager:
    def __init__(self):
        self._queue: "queue.Queue[Job]" = queue.Queue()
        self._jobs: dict[str, Job] = {}
        self._active_loader: Optional[ModelLoader] = None
        self._active_model_id: Optional[str] = None
        self._cancel_flags: set[str] = set()
        self._lock = threading.Lock()
        # Wird von der API gesetzt: Callable[[dict], None] — threadsicher.
        self.broadcast: Callable[[dict], None] = lambda event: None
        self._worker = threading.Thread(target=self._run, daemon=True, name="vid4me-worker")
        self._worker.start()

    # ---- Public API ----------------------------------------------------------

    def submit(self, req: GenerationRequest) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], request=req)
        with self._lock:
            self._jobs[job.id] = job
        self._queue.put(job)
        self._emit(job)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def list(self, limit: int = 50) -> list[dict]:
        jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        return [j.public() for j in jobs[:limit]]

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job is None:
            return False
        if job.status == "queued":
            job.status = "cancelled"
            job.finished_at = time.time()
            self._emit(job)
            return True
        if job.status in ("loading", "running"):
            self._cancel_flags.add(job_id)
            return True
        return False

    def unload_model(self) -> None:
        with self._lock:
            if self._active_loader is not None:
                self._active_loader.unload()
                self._active_loader = None
                self._active_model_id = None

    @property
    def loaded_model_id(self) -> Optional[str]:
        return self._active_model_id

    # ---- Worker ---------------------------------------------------------------

    def _emit(self, job: Job) -> None:
        self.broadcast({"type": "job", "job": job.public()})

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            if job.status == "cancelled":
                continue
            try:
                self._process(job)
            except InterruptedError:
                job.status = "cancelled"
                job.message = "Abgebrochen"
                job.finished_at = time.time()
                self._emit(job)
            except Exception as exc:  # noqa: BLE001 — Fehler gehoeren ins Job-Objekt
                job.status = "error"
                job.error = f"{type(exc).__name__}: {exc}"
                job.finished_at = time.time()
                traceback.print_exc()
                self._emit(job)
            finally:
                self._cancel_flags.discard(job.id)
                self._cleanup_after_job()

    @staticmethod
    def _cleanup_after_job() -> None:
        """RAM/VRAM-Reste nach jedem Job freigeben — bei mehreren
        Generierungen hintereinander sammelt sich sonst Speicher an."""
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _check_cancel(self, job: Job) -> None:
        if job.id in self._cancel_flags:
            raise InterruptedError

    def _process(self, job: Job) -> None:
        from . import config
        req = job.request

        def progress_cb(step: int, total: int, message: str) -> None:
            self._check_cancel(job)
            if job.status == "loading":
                # Ladephase: unbestimmter Fortschritt (nur Statustext),
                # sonst zeigt die UI faelschlich "100%" vor der Generierung.
                job.step, job.total_steps, job.progress = 0, 0, 0.0
            else:
                job.step, job.total_steps = step, total
                job.progress = step / total if total else 0.0
            job.message = message
            self._emit(job)

        # Modell laden (mit Wechsel-Logik: nur ein Modell im VRAM)
        job.status = "loading"
        job.message = "Modell wird geladen…"
        self._emit(job)

        with self._lock:
            if self._active_model_id != req.model_id:
                if self._active_loader is not None:
                    self._active_loader.unload()
                    self._active_loader = None
                    self._active_model_id = None
                settings = config.load_settings()
                loader = registry.create_loader(req.model_id, settings)
                loader.load(progress_cb)
                self._active_loader = loader
                self._active_model_id = req.model_id
            loader = self._active_loader

        self._check_cancel(job)
        job.status = "running"
        job.step, job.total_steps, job.progress = 0, 0, 0.0
        job.message = "Generierung laeuft… (erster Schritt kann einige Minuten dauern)"
        self._emit(job)

        result = loader.generate(req, progress_cb)

        job.status = "done"
        job.progress = 1.0
        job.output_path = result.output_path
        job.seed = result.seed
        job.message = "Fertig"
        job.finished_at = time.time()
        self._emit(job)


jobs = JobManager()
