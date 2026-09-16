"""LTX-2.3 Video-Plugin (Lightricks, DiT mit synchronem Audio).

Deckt die diffusers-LTX2-Pipelines ab (ab diffusers 0.37/0.38):
- LTX2Pipeline               -> Text-to-Video (+ Audio)
- LTX2ImageToVideoPipeline   -> Image-to-Video (+ Audio)

Besonderheit gegenueber Wan 2.2: LTX-2.3 erzeugt SYNCHRONES AUDIO gleich mit.
Der Export laeuft ueber diffusers.pipelines.ltx2.export_utils.encode_video
(braucht PyAV / `pip install av`); fehlt PyAV, wird stumm als reines Video
exportiert.

Ziel-Modell des Nutzers ist der 10Eros-I2V-Finetune. Der liegt als
Einzeldatei-Checkpoint (ComfyUI-Format) vor — dieser Pfad ist hier
vorbereitet (format == "single_file"), aber noch nicht scharf geschaltet
(braucht die echten Gewichte zum Verifizieren der Key-Zuordnung). Der
saubere Diffusers-Ordner-Weg (diffusers/LTX-2.3-Diffusers und der distillierte
Ableger) ist voll funktionsfaehig.
"""
from __future__ import annotations

import inspect
from typing import Optional

from backend.core.loader import (GenerationRequest, GenerationResult,
                                 LoraSpec, ModelLoader, ProgressCallback)
from . import _diffusers_common as dc

# LTX-2.3-Empfehlungen (diffusers-Doku, api/pipelines/ltx2):
# spatio-temporal-guidance auf Transformer-Block 28 (bei LTX-2.0 war es 29).
_STG_BLOCKS = [28]
# Audio-Guidance/Modality-Defaults fuer LTX-2.3 (Video-Guidance kommt aus der UI).
_DEFAULTS = {
    "stg_scale": 1.0,
    "modality_scale": 3.0,
    "audio_guidance_scale": 7.0,
    "audio_modality_scale": 3.0,
}


def _valid_num_frames(n: int) -> int:
    """LTX-2 verlangt num_frames = 8k + 1 (Default 121 = 8*15+1)."""
    n = max(9, n)
    return ((n - 1) // 8) * 8 + 1


def _round32(x: int) -> int:
    """Breite/Hoehe muessen durch 32 teilbar sein."""
    return max(32, int(round(x / 32)) * 32)


class Ltx2Loader(ModelLoader):
    plugin_name = "ltx2"

    def __init__(self, model_entry: dict, settings: dict):
        super().__init__(model_entry, settings)
        self._active_adapters: list[str] = []
        self._is_fp8 = False

    # ---- Laden ---------------------------------------------------------------

    def load(self, progress_cb: Optional[ProgressCallback] = None) -> None:
        from diffusers import LTX2ImageToVideoPipeline, LTX2Pipeline

        task = self.entry.get("task", "t2v")
        cls = LTX2ImageToVideoPipeline if task == "i2v" else LTX2Pipeline

        if self.entry.get("format") == "single_file":
            raise RuntimeError(
                "10Eros/LTX-2.3-Einzeldatei-Checkpoints werden noch nicht "
                "unterstuetzt — bitte die Diffusers-Ordner-Variante des Modells "
                "verwenden (models/video/<modell> mit model_index.json).")

        source = dc.model_source(self.entry)
        dtype = dc.resolve_dtype(self.settings)

        quant_config = None
        if self.settings.get("inference", {}).get("fp8_quantization", True):
            quant_config = self._fp8_quant_config()

        if progress_cb:
            mode = "FP8-quantisiert" if quant_config is not None else "BF16"
            progress_cb(0, 1, f"Lade LTX-2.3 ({task.upper()}, {mode})… "
                              "(erster Start kann mehrere Minuten dauern)")

        try:
            self.pipe = cls.from_pretrained(
                source, torch_dtype=dtype, quantization_config=quant_config)
            self._is_fp8 = quant_config is not None
        except Exception:
            if quant_config is None:
                raise
            if progress_cb:
                progress_cb(0, 1, "FP8 fehlgeschlagen — lade unquantisiert (BF16)…")
            self.pipe = cls.from_pretrained(source, torch_dtype=dtype)
            self._is_fp8 = False

        # LTX-2.3 (19B) + Gemma-Text-Encoder passen nur mit Offload in 24 GB.
        settings = dict(self.settings)
        inference = dict(settings.get("inference", {}))
        if inference.get("offload", "model") == "none":
            inference["offload"] = "model"
        settings["inference"] = inference
        dc.apply_offload(self.pipe, settings)
        if progress_cb:
            progress_cb(1, 1, "LTX-2.3 geladen")

    def _fp8_quant_config(self):
        """PipelineQuantizationConfig (torchao float8) fuer den Transformer;
        None, wenn torchao/diffusers-Support fehlt."""
        try:
            import torchao  # noqa: F401
            from diffusers import PipelineQuantizationConfig
        except ImportError:
            try:
                from diffusers.quantizers import PipelineQuantizationConfig  # noqa: F811
            except ImportError:
                return None
            try:
                import torchao  # noqa: F401, F811
            except ImportError:
                return None
        try:
            return PipelineQuantizationConfig(
                quant_backend="torchao",
                quant_kwargs={"quant_type": "float8wo_e4m3"},
                components_to_quantize=["transformer"],
            )
        except Exception:
            return None

    def _fit_to_image(self, image, max_area: int) -> tuple[int, int]:
        """I2V-Aufloesung aus dem Startbild-Seitenverhaeltnis ableiten
        (Flaeche ~ max_area, gerundet auf 32er-Raster)."""
        aspect = image.height / image.width
        height = _round32(int(round((max_area * aspect) ** 0.5)))
        width = _round32(int(round((max_area / aspect) ** 0.5)))
        return width, height

    # ---- LoRA-Stacking ---------------------------------------------------------

    def apply_loras(self, loras: list[LoraSpec],
                    progress_cb: Optional[ProgressCallback] = None) -> None:
        pipe = self.pipe
        # IMMER zuerst aufraeumen — auch Reste frueherer FEHLGESCHLAGENER
        # Ladeversuche (halb injizierte Adapter vergiften sonst jede weitere
        # Generierung mit diesem Modell).
        try:
            pipe.unload_lora_weights()
        except Exception:  # noqa: BLE001 — nichts geladen ist ok
            pass
        self._active_adapters = []
        if not loras:
            return

        names, weights = [], []
        try:
            for i, spec in enumerate(loras):
                path = str(dc.lora_file_path("video", spec))
                adapter = f"{dc.sanitize_adapter_name(spec.file)}_{i}"
                if progress_cb:
                    progress_cb(i, len(loras), f"Lade LoRA {spec.file}…")
                pipe.load_lora_weights(path, adapter_name=adapter)
                names.append(adapter)
                weights.append(spec.strength)
            if names:
                pipe.set_adapters(names, adapter_weights=weights)
                if self._is_fp8:
                    self._fix_lora_dtypes()
                self._active_adapters = names
        except Exception:
            # Rollback: halb geladene Adapter restlos entfernen.
            try:
                pipe.unload_lora_weights()
            except Exception:  # noqa: BLE001
                pass
            self._active_adapters = []
            raise

    def _fix_lora_dtypes(self) -> None:
        """Bei FP8-Storage legt PEFT die LoRA-Gewichte in FP8 an, womit CUDA
        nicht rechnen kann (addmm_cuda not implemented). Auf Compute-dtype
        anheben (LoRA-Gewichte sind klein)."""
        import torch

        dtype = dc.resolve_dtype(self.settings)
        tr = getattr(self.pipe, "transformer", None)
        if tr is None:
            return
        for name, p in tr.named_parameters():
            if "lora_" in name and p.dtype == torch.float8_e4m3fn:
                p.data = p.data.to(dtype)

    # ---- Generierung -------------------------------------------------------------

    def generate(self, req: GenerationRequest,
                 progress_cb: Optional[ProgressCallback] = None) -> GenerationResult:
        self.apply_loras(req.loras, progress_cb)
        seed, generator = dc.prepare_seed(req)

        num_frames = _valid_num_frames(req.num_frames)
        frame_rate = float(req.fps) if req.fps else 24.0
        width, height = _round32(req.width), _round32(req.height)

        steps = req.steps
        guidance = req.guidance_scale
        extra = req.extra or {}
        stg_scale = float(extra.get("stg_scale", _DEFAULTS["stg_scale"]))
        modality_scale = float(extra.get("modality_scale", _DEFAULTS["modality_scale"]))
        audio_guidance = extra.get("audio_guidance_scale", _DEFAULTS["audio_guidance_scale"])
        audio_modality = extra.get("audio_modality_scale", _DEFAULTS["audio_modality_scale"])

        if req.lightning:
            # Distillierter Ableger: wenige Schritte, CFG aus, keine STG/Audio-CFG.
            defaults = self.settings.get("defaults", {}).get("video", {})
            steps = min(steps, int(defaults.get("lightning_steps", 8) or 8)) or 8
            guidance = float(defaults.get("lightning_guidance_scale", 1.0))
            stg_scale = 0.0
            audio_guidance = None

        kwargs = dict(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt or None,
            width=width,
            height=height,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=steps,
            guidance_scale=guidance,
            stg_scale=stg_scale,
            modality_scale=modality_scale,
            audio_guidance_scale=audio_guidance,
            audio_modality_scale=audio_modality,
            generator=generator,
            output_type="np",
            return_dict=False,
            callback_on_step_end=dc.make_step_callback(steps, progress_cb),
        )
        if stg_scale and stg_scale > 0:
            kwargs["spatio_temporal_guidance_blocks"] = _STG_BLOCKS

        if self.entry.get("task") == "i2v":
            if not req.image_path:
                raise ValueError("I2V-Modell benoetigt ein Startbild.")
            from PIL import Image
            image = Image.open(req.image_path).convert("RGB")
            # Seitenverhaeltnis des Startbilds beibehalten (sonst Verzerrung):
            # gewaehlte Aufloesung = Pixel-Budget, Breite/Hoehe daraus berechnet.
            width, height = self._fit_to_image(image, width * height)
            image = image.resize((width, height), Image.LANCZOS)
            kwargs["image"] = image
            kwargs["width"] = width
            kwargs["height"] = height

        if req.audio_path:
            raise ValueError(
                "Audio-INPUT (Vertonung eines vorgegebenen Tons) ist bei LTX-2.3 "
                "nicht vorgesehen — das Modell erzeugt den Ton selbst passend zum Video.")

        # Pipeline liefert (video, audio) als numpy/torch.
        video, audio = self.pipe(**kwargs)

        path = dc.output_path(req, ".mp4")
        if progress_cb:
            progress_cb(steps, steps, "Exportiere Video…")
        has_audio = self._export(video[0], audio, frame_rate, path)

        dc.write_sidecar(path, req, seed,
                         extra={"num_frames": num_frames, "frame_rate": frame_rate,
                                "effective_steps": steps, "has_audio": has_audio,
                                "output_width": kwargs["width"],
                                "output_height": kwargs["height"]})
        return GenerationResult(output_path=str(path), seed=seed,
                                metadata={"num_frames": num_frames, "has_audio": has_audio})

    def _export(self, video, audio, frame_rate: float, path) -> bool:
        """Video mit synchronem Audio schreiben (PyAV). Fehlt PyAV oder Audio,
        wird stumm exportiert. Gibt zurueck, ob Audio geschrieben wurde."""
        audio_wave = None
        if audio is not None:
            try:
                audio_wave = audio[0].float().cpu()
            except (AttributeError, IndexError, TypeError):
                audio_wave = None

        if audio_wave is not None:
            try:
                from diffusers.pipelines.ltx2.export_utils import encode_video
                vocoder = getattr(self.pipe, "vocoder", None)
                sr = int(getattr(getattr(vocoder, "config", None),
                                 "output_sampling_rate", 24000))
                encode_video(video, fps=int(round(frame_rate)), audio=audio_wave,
                             audio_sample_rate=sr, output_path=str(path))
                return True
            except ImportError:
                # PyAV nicht installiert -> stummer Fallback
                pass
            except Exception:
                pass

        # Fallback: reines Video (numpy [0,1] -> uint8-Frames)
        import numpy as np
        from diffusers.utils import export_to_video
        frames = [(np.clip(f, 0, 1) * 255).round().astype("uint8") for f in video]
        export_to_video(frames, str(path), fps=int(round(frame_rate)))
        return False


LOADER = Ltx2Loader
