"""Abstraktes Modell-Loader-Interface.

Jede Modell-Familie wird als Plugin in backend/plugins/ implementiert und
leitet von ModelLoader ab. Der Core kennt nur dieses Interface — neue
Modelle brauchen keine Aenderung am Core-Code.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# progress_cb(step, total_steps, message)
ProgressCallback = Callable[[int, int, str], None]


@dataclass
class LoraSpec:
    """Ein Eintrag im LoRA-Stack, wie vom Frontend uebergeben."""
    file: str                 # Dateiname relativ zum LoRA-Ordner des Modelltyps
    strength: float = 1.0
    # Nur fuer Wan 2.2 MoE relevant: auf welchen Experten anwenden.
    # "both" | "high" | "low"  (Bildmodelle ignorieren das Feld)
    target: str = "both"


@dataclass
class GenerationRequest:
    model_id: str
    prompt: str
    negative_prompt: str = ""
    seed: int = -1                       # -1 = zufaellig
    steps: int = 40
    guidance_scale: float = 4.0
    width: int = 1024
    height: int = 1024
    # Video-spezifisch
    num_frames: int = 81
    fps: int = 16
    image_path: Optional[str] = None     # Startbild fuer I2V
    audio_path: Optional[str] = None     # reserviert fuer S2V (noch ohne Diffusers-Support)
    lightning: bool = False              # 4-8-Schritt-Distillation-Modus
    loras: list[LoraSpec] = field(default_factory=list)
    project: str = "default"             # Unterordner in outputs/
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    output_path: str
    seed: int
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelLoader(ABC):
    """Basisklasse fuer alle Modell-Plugins.

    Lebenszyklus: instanziert pro Registry-Eintrag, load() beim ersten
    Generieren, unload() wenn ein anderes Modell VRAM braucht.
    """

    # Von Subklassen ueberschrieben
    plugin_name: str = "base"

    def __init__(self, model_entry: dict, settings: dict):
        self.entry = model_entry          # Registry-Eintrag (id, repo_id, local_path, ...)
        self.settings = settings
        self.pipe = None

    # ---- Pflicht-Interface -------------------------------------------------

    @abstractmethod
    def load(self, progress_cb: Optional[ProgressCallback] = None) -> None:
        """Pipeline in Speicher/VRAM laden (inkl. Offload-Strategie)."""

    @abstractmethod
    def generate(self, req: GenerationRequest,
                 progress_cb: Optional[ProgressCallback] = None) -> GenerationResult:
        """Eine Generierung ausfuehren. Muss LoRA-Stack aus req.loras anwenden."""

    # ---- Optionales Interface ----------------------------------------------

    def apply_loras(self, loras: list[LoraSpec],
                    progress_cb: Optional[ProgressCallback] = None) -> None:
        """Generisches LoRA-Stacking. Subklassen ueberschreiben bei Bedarf."""
        raise NotImplementedError(f"{self.plugin_name} unterstuetzt keine LoRAs")

    def unload(self) -> None:
        """Pipeline freigeben und VRAM aufraeumen."""
        self.pipe = None
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    @property
    def is_loaded(self) -> bool:
        return self.pipe is not None

    # ---- Faehigkeiten fuer die UI -------------------------------------------

    def capabilities(self) -> list[str]:
        return list(self.entry.get("capabilities", []))

    def supports_audio(self) -> bool:
        return "audio_sync" in self.capabilities()
