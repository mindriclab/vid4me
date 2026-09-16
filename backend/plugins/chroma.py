"""Chroma1-HD Bild-Plugin (Apache 2.0, basiert auf Flux.1-Schnell, 8.9B)."""
from __future__ import annotations

from typing import Optional

from backend.core.loader import (GenerationRequest, GenerationResult,
                                 LoraSpec, ModelLoader, ProgressCallback)
from . import _diffusers_common as dc


class ChromaLoader(ModelLoader):
    plugin_name = "chroma"

    def __init__(self, model_entry: dict, settings: dict):
        super().__init__(model_entry, settings)
        self._active_adapters: list[str] = []

    def load(self, progress_cb: Optional[ProgressCallback] = None) -> None:
        from diffusers import ChromaPipeline

        if progress_cb:
            progress_cb(0, 1, "Lade Chroma1-HD…")
        self.pipe = ChromaPipeline.from_pretrained(
            dc.model_source(self.entry),
            torch_dtype=dc.resolve_dtype(self.settings),
        )
        dc.apply_offload(self.pipe, self.settings)
        if progress_cb:
            progress_cb(1, 1, "Chroma1-HD geladen")

    def apply_loras(self, loras: list[LoraSpec],
                    progress_cb: Optional[ProgressCallback] = None) -> None:
        """Generisches LoRA-Stacking ueber Diffusers-Adapter."""
        pipe = self.pipe
        # Vorherigen Stack deaktivieren
        if self._active_adapters:
            pipe.unload_lora_weights()
            self._active_adapters = []
        names, weights = [], []
        for i, spec in enumerate(loras):
            path = dc.lora_file_path("image", spec)
            adapter = f"{dc.sanitize_adapter_name(spec.file)}_{i}"
            if progress_cb:
                progress_cb(i, len(loras), f"Lade LoRA {spec.file}…")
            pipe.load_lora_weights(str(path), adapter_name=adapter)
            names.append(adapter)
            weights.append(spec.strength)
        if names:
            pipe.set_adapters(names, adapter_weights=weights)
        self._active_adapters = names

    def generate(self, req: GenerationRequest,
                 progress_cb: Optional[ProgressCallback] = None) -> GenerationResult:
        self.apply_loras(req.loras, progress_cb)
        seed, generator = dc.prepare_seed(req)

        image = self.pipe(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt or None,
            width=req.width,
            height=req.height,
            num_inference_steps=req.steps,
            guidance_scale=req.guidance_scale,
            generator=generator,
            callback_on_step_end=dc.make_step_callback(req.steps, progress_cb),
        ).images[0]

        path = dc.output_path(req, ".png")
        image.save(path)
        dc.write_sidecar(path, req, seed)
        return GenerationResult(output_path=str(path), seed=seed)


LOADER = ChromaLoader
