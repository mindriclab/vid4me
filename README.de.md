# Vid4me

Lokale Desktop-App für Video- **und** Bildgenerierung — komplett offline, ohne Cloud-APIs.
Electron/React-Frontend + Python/FastAPI-Backend, modulare Modell-Plugins, generisches LoRA-Stacking.

## Modelle

| Typ | Modell | Diffusers-Pipeline | Quelle |
|---|---|---|---|
| Video (T2V) | Wan 2.2 T2V **A14B** (MoE, high+low noise) | `WanPipeline` | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` |
| Video (I2V) | Wan 2.2 I2V **A14B** (MoE) | `WanImageToVideoPipeline` | `Wan-AI/Wan2.2-I2V-A14B-Diffusers` |
| Bild | Chroma1-HD (8.9B, Apache 2.0) | `ChromaPipeline` | `lodestones/Chroma1-HD` |

Bewusst **nicht** verwendet: die TI2V-5B-Variante (deutlich schwächere Ergebnisse laut Community).

## Installation auf einem anderen Rechner (Laptop etc.)

```powershell
cd frontend
npm run dist
```

erzeugt in `frontend/release/`:

- **`Vid4me Setup 0.1.0.exe`** — klassischer Windows-Installer (Startmenü, Desktop-Icon)
- **`Vid4me-0.1.0-portable.exe`** — portable Einzeldatei, ohne Installation startbar

Beim **ersten Start** richtet sich die App selbst ein: Sie legt den Datenordner
`%USERPROFILE%\Vid4me` an (Modelle, Ergebnisse, Einstellungen, Python-Umgebung)
und installiert die Python-Umgebung automatisch (~3 GB; findet sie kein Python ≥ 3.10
auf dem System, lädt sie ein portables Python 3.12 selbst herunter).
Danach Modelle direkt in der App über **„⬇ Modelle"** herunterladen — sie landen
automatisch im richtigen Ordner.

## Entwicklung (dieser Projektordner)

```powershell
# einmalig: Python-Backend (venv, FastAPI, torch CUDA 12.8, diffusers, ~3 GB)
powershell -ExecutionPolicy Bypass -File scripts\setup_backend.ps1
cd frontend && npm install

# starten
npm run dev      # Entwicklung (Vite + Electron, Hot-Reload)
npm start        # baut das UI und startet Electron
```

Die Electron-App startet das Python-Backend **automatisch** (Port 8756, konfigurierbar
in `config/settings.json`) und beendet es beim Schließen wieder. Manueller Backend-Start
für Entwicklung: `.venv\Scripts\python -m backend.main`

## Modelle verwalten (in der App: „⬇ Modelle")

- **Herunterladen**: Modell auswählen → Download läuft mit Live-Fortschritt (GB-Anzeige),
  abbrechbar, jederzeit fortsetzbar (bereits geladene Dateien werden wiederverwendet).
  Beschleunigt durch `hf_transfer`. Größen: Chroma ~26 GB, Wan-Varianten je ~118 GB
  (2× 14B-Experten in FP32 + Text-Encoder) — Festplattenplatz einplanen!
- **Löschen**: „🗑 Dateien" gibt den Speicher eines Modells wieder frei
  (Eintrag bleibt, Neu-Download jederzeit möglich). LoRAs: 🗑 im LoRA-Panel.
- **Lokale Modelle einbinden**: Vorhandene Modelle einfach nach `models/video/` bzw.
  `models/image/` kopieren → sie erscheinen im Modell-Manager unter „Lokales Modell
  einbinden". Unterstützt werden **Diffusers-Ordner** und **Einzeldatei-Checkpoints**
  (.safetensors, ComfyUI-Format, nur Wan-Video). Alternativ beliebigen Ordnerpfad
  angeben (Modell bleibt, wo es ist).
- **Einzeldatei-Checkpoints (ComfyUI)**: Solche Dateien enthalten nur den
  Diffusion-Transformer. VAE, Text-Encoder, Tokenizer und Scheduler kommen aus dem
  einmalig ladbaren Paket **„Wan 2.2 Basis-Komponenten" (~11 GB)** im Modell-Manager.
  Wan 2.2 A14B kommt meist als **Paar** (High-/Low-Noise-Datei) — beim Einbinden beide
  angeben; merged Checkpoints (eine Datei) laufen mit einem Transformer.
  Hinweis: ComfyUI-„fp8_scaled"-Varianten kann Diffusers nicht laden — BF16/FP16-Version
  verwenden (FP8-Quantisierung übernimmt die App selbst via torchao).
- **Hugging-Face-Suche**: Modelle suchen und mit Typ/Plugin/Aufgabe zur Registry hinzufügen.

## VRAM-Strategie (Wan 2.2 14B, fester Standard)

FP16 bräuchte 55–65 GB VRAM. Deshalb lädt das wan22-Plugin standardmäßig:
**FP8-Quantisierung (torchao)** für beide Experten-Transformer **+ CPU-Offload**
(Text-Encoder in den System-RAM) → ~14–16 GB VRAM auf der RTX 4090.
Fallback auf BF16+Offload, falls torchao fehlt. Konfigurierbar in
`config/settings.json` (`inference.fp8_quantization`), `offload: "none"` wird für
Wan bewusst ignoriert.

## LoRA-System (generisch, neutral)

- `.safetensors`-Dateien einfach in `models/loras/video/` bzw. `models/loras/image/` legen
  (oder über die UI hochladen) — sie werden automatisch erkannt und in
  `models/registry.json` mit Metadaten (Name, Standard-Stärke, Ziel) registriert.
- Mehrere LoRAs gleichzeitig aktivierbar, jede mit eigenem Stärke-Regler (0.0–1.5+).
- Kombinationen als frei benennbare **Presets** speicherbar.
- Wan 2.2 MoE: pro LoRA wählbar, ob sie auf **beide Experten**, nur den
  **High-Noise**- oder nur den **Low-Noise**-Transformer wirkt.
- **Lightning-Modus** (4–8 Schritte): passende Distillation-LoRAs (z. B.
  Wan2.2-Lightning, als getrennte High/Low-Dateien) in den Stack legen, High-Datei auf
  „nur High-Noise", Low-Datei auf „nur Low-Noise" stellen, Lightning-Toggle aktivieren —
  CFG wird automatisch auf 1.0 gesetzt. Die App selbst liefert keine LoRAs mit.

## Architektur

```
vid4me/
├── models/               registry.json + Gewichte + LoRA-Ordner
├── outputs/<projekt>/    Ergebnisse (+ JSON-Sidecar mit allen Parametern/Seed)
├── config/settings.json  Port, Offload-Strategie, Defaults
├── backend/
│   ├── main.py           FastAPI-Einstieg (REST + WebSocket /ws)
│   ├── core/             loader.py (abstraktes Plugin-Interface), registry.py,
│   │                     jobs.py (serielle GPU-Queue), vram.py, config.py
│   ├── api/              routes.py, ws.py
│   └── plugins/          wan22.py, chroma.py — neue Modell-Familie = neues Modul
│                         mit `LOADER = <ModelLoader-Subklasse>`, kein Core-Umbau
└── frontend/             Electron (startet Backend) + React/Vite-UI
```

**Plugin-Interface** (`backend/core/loader.py`): `load()`, `generate(request, progress_cb)`,
`apply_loras(stack)`, `unload()`. Die Registry entdeckt Plugins automatisch per
`pkgutil` über den `plugins/`-Ordner.

## VRAM (RTX 4090, 24 GB)

- Standard: `enable_model_cpu_offload()` + VAE-Tiling (in `config/settings.json`:
  `inference.offload` = `model` | `sequential` | `none`).
- Die UI zeigt den VRAM-Status live an und warnt vor der Generierung, wenn zu wenig
  Speicher frei ist.
- FP8-Quantisierung ist als Option vorgesehen (`inference.fp8_quantization`), aktuell
  noch nicht implementiert.

## Audio-Sync (Stand der Recherche, Juli 2026)

- **Wan2.2-S2V-14B** (audio-getriebene Videogenerierung inkl. Lip-Sync) existiert als
  offenes Modell mit Checkpoints, ist aber **noch nicht in Diffusers integriert**
  ([Issue #12257](https://github.com/huggingface/diffusers/issues/12257)).
- Das Audio-Upload-Feld in der UI ist deshalb vorbereitet, aber deaktiviert. Sobald die
  Diffusers-Integration (oder eine eigene S2V-Pipeline-Portierung) verfügbar ist, wird
  sie als neues Plugin (`backend/plugins/wan22_s2v.py`) ergänzt — ohne Core-Änderung.
- Lokales TTS als Vorstufe bleibt als späteres, separates Modul eingeplant.
