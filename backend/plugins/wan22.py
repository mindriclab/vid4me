"""Wan 2.2 A14B Video-Plugin (T2V + I2V, MoE mit High/Low-Noise-Experten).

Bewusst NUR die 14B-MoE-Variante (transformer + transformer_2,
boundary_ratio-gesteuerter Experten-Wechsel). LoRAs koennen gezielt auf den
High-Noise-Experten (transformer), den Low-Noise-Experten (transformer_2)
oder beide angewendet werden — noetig z.B. fuer Lightning-LoRAs, die als
getrennte High/Low-Dateien ausgeliefert werden.
"""
from __future__ import annotations

import inspect
from typing import Optional

from backend.core.loader import (GenerationRequest, GenerationResult,
                                 LoraSpec, ModelLoader, ProgressCallback)
from . import _diffusers_common as dc


def _valid_num_frames(n: int) -> int:
    """Wan verlangt num_frames = 4k + 1."""
    n = max(5, n)
    return ((n - 1) // 4) * 4 + 1


class Wan22Loader(ModelLoader):
    plugin_name = "wan22"

    def __init__(self, model_entry: dict, settings: dict):
        super().__init__(model_entry, settings)
        self._has_lora_stack = False

    # ---- Laden ---------------------------------------------------------------

    def load(self, progress_cb: Optional[ProgressCallback] = None) -> None:
        from diffusers import WanImageToVideoPipeline, WanPipeline

        task = self.entry.get("task", "t2v")
        cls = WanImageToVideoPipeline if task == "i2v" else WanPipeline
        is_single_file = self.entry.get("format") == "single_file"
        source = None if is_single_file else dc.model_source(self.entry)
        dtype = dc.resolve_dtype(self.settings)

        # FESTER STANDARD fuer 24-GB-Karten (siehe VID4ME_PROJECT_v4.md):
        # FP8-Quantisierung der beiden 14B-Experten-Transformer (torchao)
        # + CPU-Offload (Text-Encoder etc. in den System-RAM).
        # FP16/BF16 ohne Quantisierung braeuchte 55-65 GB VRAM.
        quant_config = None
        if self.settings.get("inference", {}).get("fp8_quantization", True):
            quant_config = self._fp8_quant_config()

        if progress_cb:
            mode = "FP8-quantisiert" if quant_config is not None else "BF16"
            progress_cb(0, 1, f"Lade Wan 2.2 ({task.upper()}, {mode})… "
                              "(erster Start kann mehrere Minuten dauern)")

        if is_single_file:
            self.pipe = self._load_single_file(cls, dtype, quant_config, progress_cb)
        else:
            try:
                self.pipe = cls.from_pretrained(
                    source, torch_dtype=dtype, quantization_config=quant_config)
            except Exception:
                if quant_config is None:
                    raise
                # Fallback: ohne Quantisierung laden (braucht mehr VRAM/Offload)
                if progress_cb:
                    progress_cb(0, 1, "FP8 fehlgeschlagen — lade unquantisiert (BF16)…")
                self.pipe = cls.from_pretrained(source, torch_dtype=dtype)

        # CPU-Offload ist bei Wan 2.2 14B Pflicht, nicht optional:
        # "none" wird bewusst auf Model-Offload angehoben.
        settings = dict(self.settings)
        inference = dict(settings.get("inference", {}))
        if inference.get("offload", "model") == "none":
            inference["offload"] = "model"
        settings["inference"] = inference
        dc.apply_offload(self.pipe, settings)
        if progress_cb:
            progress_cb(1, 1, "Wan 2.2 geladen")

    def _load_single_file(self, cls, dtype, quant_config, progress_cb):
        """Pipeline aus Einzeldatei-Checkpoint(s) (ComfyUI-Format) bauen.

        Die Einzeldatei enthaelt nur den Diffusion-Transformer. VAE,
        UMT5-Text-Encoder, Tokenizer und Scheduler kommen aus dem
        Basis-Komponenten-Paket (Registry-Eintrag wan22-components).
        - high + low Datei  -> echtes MoE (transformer + transformer_2)
        - nur eine Datei    -> merged Checkpoint, ein Transformer fuer alle Steps
        """
        import torch
        from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanTransformer3DModel
        from transformers import AutoTokenizer, UMT5EncoderModel
        from backend.core import config
        from backend.core.registry import registry

        comp_entry = registry.get_model("wan22-components")
        comp = config.resolve(comp_entry["local_path"]) if comp_entry else None
        if comp is None or not (comp / "model_index.json").exists():
            raise RuntimeError(
                "Basis-Komponenten fehlen. Bitte im Modell-Manager "
                "'Wan 2.2 Basis-Komponenten (VAE, Text-Encoder)' herunterladen (~11 GB) — "
                "Einzeldatei-Checkpoints enthalten nur den Transformer.")

        files = self.entry.get("single_files", {})
        high = config.resolve(files["high"])
        low = config.resolve(files["low"]) if files.get("low") else None

        transformer = self._load_single_transformer(
            high, comp, "transformer", dtype,
            "High-Noise" if low else "merged", progress_cb)
        transformer_2 = (self._load_single_transformer(
            low, comp, "transformer_2", dtype, "Low-Noise", progress_cb)
            if low else None)

        if progress_cb:
            progress_cb(0, 1, "Lade Basis-Komponenten (VAE, Text-Encoder)…")
        self._require_ram(14, "den UMT5-Text-Encoder (~11 GB)")
        components = {
            "transformer": transformer,
            "text_encoder": UMT5EncoderModel.from_pretrained(
                str(comp), subfolder="text_encoder", torch_dtype=dtype),
            "tokenizer": AutoTokenizer.from_pretrained(str(comp), subfolder="tokenizer"),
            "vae": AutoencoderKLWan.from_pretrained(
                str(comp), subfolder="vae", torch_dtype=torch.float32),
            "scheduler": UniPCMultistepScheduler.from_pretrained(
                str(comp), subfolder="scheduler"),
            # optionale/legacy Parameter mancher Pipeline-Versionen
            "image_encoder": None,
            "image_processor": None,
            "transformer_2": transformer_2,
            "boundary_ratio": (0.9 if self.entry.get("task") == "i2v" else 0.875)
                              if transformer_2 is not None else None,
        }
        sig = inspect.signature(cls.__init__).parameters
        kwargs = {k: v for k, v in components.items() if k in sig}
        if transformer_2 is not None and "transformer_2" not in sig:
            raise RuntimeError("Diese diffusers-Version unterstuetzt kein "
                               "High/Low-Paar (transformer_2) — bitte aktualisieren.")
        return cls(**kwargs)

    def _fit_to_image(self, image, max_area: int) -> tuple[int, int]:
        """Video-Aufloesung aus dem Startbild-Seitenverhaeltnis ableiten
        (Flaeche ~ max_area, gerundet auf das Modell-Raster)."""
        try:
            mod = (self.pipe.vae_scale_factor_spatial
                   * self.pipe.transformer.config.patch_size[1])
        except (AttributeError, TypeError, IndexError):
            mod = 16
        aspect = image.height / image.width
        height = max(mod, int(round((max_area * aspect) ** 0.5)) // mod * mod)
        width = max(mod, int(round((max_area / aspect) ** 0.5)) // mod * mod)
        return width, height

    # ---- Einzeldatei: Konvertierung + speicherschonendes Laden ----------------

    @staticmethod
    def _require_ram(needed_gb: float, what: str) -> None:
        """Harte RAM-Pruefung VOR grossen Allokationen — verhindert System-Freeze."""
        import psutil
        avail = psutil.virtual_memory().available / 1024**3
        if avail < needed_gb:
            raise RuntimeError(
                f"Zu wenig freier Arbeitsspeicher fuer {what}: "
                f"{avail:.1f} GB frei, ~{needed_gb:.0f} GB benoetigt. "
                "Bitte andere Programme schliessen und erneut versuchen.")

    def _load_single_transformer(self, path, comp, config_subdir, dtype,
                                 label, progress_cb):
        """Einzeldatei-Checkpoint laden.

        Schritt 1 (einmalig pro Datei): Konvertierung in einen FP8-Storage-
        Ordner (models/video/.fp8cache/<name>/) in einem SUBPROZESS — der
        braucht kurzzeitig viel RAM (~3x Dateigroesse), gibt ihn danach aber
        vollstaendig zurueck. Ein Watchdog bricht ab, bevor der RAM ausgeht.

        Schritt 2 (jeder Start): Cache memory-mapped mit dtype-Erhalt laden
        (nahezu kein RAM-Bedarf) + Layerwise-Casting (Storage FP8,
        Compute BF16) — passt in 24 GB VRAM.
        """
        import gc
        import re

        import safetensors.torch
        import torch
        from accelerate import init_empty_weights
        from diffusers import WanTransformer3DModel
        from backend.core import config

        cache_dir = (config.MODELS_DIR / "video" / ".fp8cache"
                     / re.sub(r"[^\w\-]", "_", path.stem))

        if not (cache_dir / "config.json").exists():
            self._convert_single_file(path, comp, config_subdir, cache_dir,
                                      label, progress_cb)

        if progress_cb:
            progress_cb(0, 1, f"Lade Transformer ({label}) aus FP8-Cache…")
        with init_empty_weights():
            model = WanTransformer3DModel.from_config(
                WanTransformer3DModel.load_config(str(cache_dir)))
        state = {}
        for shard in sorted(cache_dir.glob("*.safetensors")):
            state.update(safetensors.torch.load_file(str(shard)))
        model.load_state_dict(state, strict=True, assign=True)
        state = None
        gc.collect()
        model.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn,
                                       compute_dtype=dtype)
        return model

    def _convert_single_file(self, path, comp, config_subdir, cache_dir,
                             label, progress_cb) -> None:
        """Einmal-Konvertierung im Subprozess (RAM wird danach komplett frei)."""
        import subprocess
        import sys

        file_gb = path.stat().st_size / 1024**3
        # Gemessen: Peak ~3x Dateigroesse (Checkpoint + Modell + Konvertierung)
        self._require_ram(file_gb * 3.0 + 6, f"die einmalige Konvertierung von {path.name}")

        cfg_dir = comp / config_subdir
        if not (cfg_dir / "config.json").exists():
            cfg_dir = comp / "transformer"

        if progress_cb:
            progress_cb(0, 1, f"Konvertiere {path.name} einmalig ins FP8-Format… "
                              f"(~{file_gb * 3:.0f} GB RAM, mehrere Minuten)")
        code = (
            "import os, threading, time, torch, psutil\n"
            "def wd():\n"
            "    while True:\n"
            "        if psutil.virtual_memory().available < 4 * 1024**3:\n"
            "            os._exit(3)\n"
            "        time.sleep(0.5)\n"
            "threading.Thread(target=wd, daemon=True).start()\n"
            "from diffusers import WanTransformer3DModel\n"
            f"m = WanTransformer3DModel.from_single_file({str(path)!r}, "
            f"config={str(cfg_dir)!r}, torch_dtype=torch.bfloat16)\n"
            "m.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, "
            "compute_dtype=torch.bfloat16)\n"
            f"m.save_pretrained({str(cache_dir)!r}, max_shard_size='10GB')\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=3600,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if proc.returncode == 3:
            raise RuntimeError(
                f"Konvertierung von {path.name} abgebrochen: Arbeitsspeicher wurde "
                "knapp (Schutzabschaltung). Bitte andere Programme schliessen.")
        if proc.returncode != 0:
            raise RuntimeError(
                f"Konvertierung von {path.name} fehlgeschlagen:\n"
                f"{(proc.stderr or '')[-600:]}\n"
                "Hinweis: ComfyUI-'fp8_scaled'-Varianten (mit Scale-Tensoren) "
                "werden nicht unterstuetzt.")
        if progress_cb:
            progress_cb(1, 1, f"Konvertierung von {path.name} abgeschlossen")

    def _fp8_quant_config(self):
        """PipelineQuantizationConfig fuer FP8 (torchao) bauen; None wenn
        torchao/diffusers-Support fehlt."""
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
                components_to_quantize=["transformer", "transformer_2"],
            )
        except Exception:
            return None

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
        self._has_lora_stack = False
        if not loras:
            return

        names, weights = [], []
        has_t2 = getattr(pipe, "transformer_2", None) is not None

        try:
            for i, spec in enumerate(loras):
                path = str(dc.lora_file_path("video", spec))
                base = f"{dc.sanitize_adapter_name(spec.file)}_{i}"
                if progress_cb:
                    progress_cb(i, len(loras), f"Lade LoRA {spec.file} ({spec.target})…")

                target = spec.target if spec.target in ("both", "high", "low") else "both"
                if target == "low" and not has_t2:
                    raise RuntimeError(
                        f"LoRA {spec.file}: Ziel 'nur Low-Noise' geht nicht — dieses "
                        "Modell hat nur einen Transformer (merged Checkpoint). "
                        "Bitte Ziel 'beide Experten' waehlen.")
                if target in ("both", "high"):
                    pipe.load_lora_weights(path, adapter_name=f"{base}_hn")
                    names.append(f"{base}_hn")
                    weights.append(spec.strength)
                if target in ("both", "low") and has_t2:
                    try:
                        pipe.load_lora_weights(path, adapter_name=f"{base}_ln",
                                               load_into_transformer_2=True)
                    except TypeError as exc:
                        raise RuntimeError(
                            "Diese diffusers-Version kennt load_into_transformer_2 "
                            "nicht — bitte diffusers aktualisieren (>= 0.36).") from exc
                    names.append(f"{base}_ln")
                    weights.append(spec.strength)

            if names:
                pipe.set_adapters(names, adapter_weights=weights)
                self._fix_lora_dtypes()
                self._has_lora_stack = True
        except Exception:
            # Rollback: halb geladene Adapter restlos entfernen, sonst ist das
            # Modell fuer alle folgenden Laeufe unbrauchbar.
            try:
                pipe.unload_lora_weights()
            except Exception:  # noqa: BLE001
                pass
            raise

    def _fix_lora_dtypes(self) -> None:
        """PEFT legt LoRA-Gewichte im dtype der Basisschicht an — bei
        FP8-Storage-Modellen also FP8, womit CUDA nicht rechnen kann
        (addmm_cuda not implemented). LoRA-Gewichte sind klein: auf das
        Compute-dtype anheben."""
        import torch

        dtype = dc.resolve_dtype(self.settings)
        for tr in (getattr(self.pipe, "transformer", None),
                   getattr(self.pipe, "transformer_2", None)):
            if tr is None:
                continue
            for name, p in tr.named_parameters():
                if "lora_" in name and p.dtype == torch.float8_e4m3fn:
                    p.data = p.data.to(dtype)

    # ---- Generierung -------------------------------------------------------------

    def generate(self, req: GenerationRequest,
                 progress_cb: Optional[ProgressCallback] = None) -> GenerationResult:
        from diffusers.utils import export_to_video

        self.apply_loras(req.loras, progress_cb)
        seed, generator = dc.prepare_seed(req)

        num_frames = _valid_num_frames(req.num_frames)
        steps = req.steps
        guidance = req.guidance_scale
        if req.lightning:
            # Distillierte 4-8-Schritt-Inferenz: CFG aus, wenige Schritte.
            defaults = self.settings.get("defaults", {}).get("video", {})
            steps = min(steps, int(defaults.get("lightning_steps", 4) or 4)) or 4
            guidance = float(defaults.get("lightning_guidance_scale", 1.0))

        kwargs = dict(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt or None,
            width=req.width,
            height=req.height,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
            callback_on_step_end=dc.make_step_callback(steps, progress_cb),
        )

        # MoE: eigener CFG-Wert fuer den Low-Noise-Experten, falls unterstuetzt
        try:
            if "guidance_scale_2" in inspect.signature(self.pipe.__call__).parameters:
                kwargs["guidance_scale_2"] = guidance
        except (TypeError, ValueError):
            pass

        if self.entry.get("task") == "i2v":
            if not req.image_path:
                raise ValueError("I2V-Modell benoetigt ein Startbild.")
            from PIL import Image
            image = Image.open(req.image_path).convert("RGB")
            # Seitenverhaeltnis des Startbilds BEIBEHALTEN (sonst Verzerrung):
            # Die gewaehlte Aufloesung dient als Pixel-Budget, Breite/Hoehe
            # werden aus dem Bild berechnet (auf Modell-Raster gerundet).
            width, height = self._fit_to_image(image, req.width * req.height)
            image = image.resize((width, height), Image.LANCZOS)
            kwargs["image"] = image
            kwargs["width"] = width
            kwargs["height"] = height

        if req.audio_path:
            raise ValueError(
                "Audio-Sync (Wan2.2-S2V) ist in diffusers noch nicht verfuegbar — "
                "das Feld ist fuer eine spaetere Plugin-Version reserviert.")

        frames = self.pipe(**kwargs).frames[0]

        path = dc.output_path(req, ".mp4")
        if progress_cb:
            progress_cb(steps, steps, "Exportiere Video…")
        export_to_video(frames, str(path), fps=req.fps)
        dc.write_sidecar(path, req, seed,
                         extra={"num_frames": num_frames, "fps": req.fps,
                                "effective_steps": steps,
                                "output_width": kwargs.get("width", req.width),
                                "output_height": kwargs.get("height", req.height)})
        return GenerationResult(output_path=str(path), seed=seed,
                                metadata={"num_frames": num_frames})


LOADER = Wan22Loader
