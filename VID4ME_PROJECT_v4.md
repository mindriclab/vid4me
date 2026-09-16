# Vid4me – Projekt-Startprompt für Claude Code

## Projektbeschreibung

Ich baue "Vid4me", eine eigenstaendige lokale Desktop-App fuer Video- 
UND Bildgenerierung, komplett offline, ohne Cloud-APIs. Ziel ist ein 
hochwertiger, lokaler Ersatz fuer ComfyUI-basierte Workflows mit einer 
modernen, professionellen Oberflaeche. Privates Projekt.

## Hardware

- RTX 4090, 24GB VRAM

## UI/App-Architektur: Electron + Python-Backend

- Frontend: Electron (React oder Vue) fuer eine moderne, professionelle 
  Desktop-Oberflaeche auf dem Niveau kommerzieller Tools
- Backend: Python (FastAPI) laeuft lokal, uebernimmt Modell-Loading und 
  Inferenz, kommuniziert mit dem Electron-Frontend ueber lokale REST-API/
  WebSocket
- Begruendung: beste erreichbare UI-Qualitaet, wichtiger als 
  Implementierungsaufwand - KEIN Gradio, KEIN reines PySide6
- Electron-App startet den Python-Backend-Prozess automatisch im 
  Hintergrund (kein manueller Serverstart durch mich noetig)

## Modell-Architektur: Modular/Austauschbar

- Modell-Registry-System nach dem Hugging Face "Modular Diffusers"-
  Prinzip, austauschbare Bausteine statt hart verdrahteter Pipeline
- UI: Dropdown zur Modellauswahl (Video/Bild getrennt gelistet)
- Neue Modelle ueber Plugin-Ordner ergaenzbar, ohne Core-Code zu aendern

## Initiale Modelle

**Video-Basismodell:** Wan 2.2 (Wan-AI/Alibaba, offen verfuegbar auf 
GitHub Wan-Video/Wan2.2 und Hugging Face Wan-AI)

- Ausschliesslich die 14B MoE-Variante verwenden (T2V-A14B / I2V-A14B, 
  high+low noise Experten-Architektur) als Standard-Zielmodell. NICHT 
  die TI2V-5B-Variante - diese liefert laut Community-Feedback deutlich 
  schwaechere Ergebnisse
- Offloading-Strategien fuer 24GB VRAM beruecksichtigen (CPU-Offload, 
  ggf. FP8-Quantisierung als Option)

**Bild:** Chroma (Apache 2.0, basiert auf Flux.1-Schnell, 8.9B Parameter)

## Generisches LoRA-Stacking-System (WICHTIG)

Die App soll ein allgemeines, offenes LoRA-Verwaltungssystem bieten, 
unabhaengig vom konkreten Anwendungszweck - vergleichbar mit dem 
LoRA-Handling in ComfyUI/A1111:

- Nutzer kann eigene LoRA-Dateien (.safetensors) ueber die UI hochladen/
  in einen lokalen Ordner legen, unabhaengig von Herkunft oder Inhalt
- Mehrere LoRAs gleichzeitig aktivierbar, jede mit individuellem 
  Staerke-Regler (0.0-1.0+)
- Speicherbare LoRA-Kombinations-Presets (frei benennbar durch den 
  Nutzer)
- Automatische Erkennung neu hinzugefuegter LoRA-Dateien im 
  entsprechenden Ordner, keine Code-Aenderung noetig
- Getrennte LoRA-Slots fuer Video- und Bildmodell
- Zusaetzlich: Self-Forcing/Lightning-LoRA-Support fuer 4-8-Schritt-
  Inferenz zur Geschwindigkeitssteigerung (Performance-Feature, 
  optional zuschaltbar, Standard bleibt volle Schrittzahl)
- Die App selbst enthaelt und empfiehlt KEINE spezifischen LoRAs - das 
  System ist neutral, die Auswahl liegt vollstaendig beim Nutzer

### Technische Anforderung an die Registry

```
models/
├── video/
│   └── wan22-14b/
├── image/
│   └── chroma/
└── loras/
    ├── video/      # vom Nutzer selbst hinzugefuegte LoRA-Dateien
    └── image/      # vom Nutzer selbst hinzugefuegte LoRA-Dateien
```

registry.json verwaltet Metadaten (Name, Staerke-Default, Modelltyp) 
pro LoRA-Datei, die der Nutzer im jeweiligen Ordner ablegt.

## Bild-zu-Video mit Startbild

- Upload-Feld fuer ein Startbild als Grundlage der Video-Generierung 
  (Standard I2V-Workflow)
- Prompt-Textfeld + negativer Prompt zur Steuerung von Bewegung/Szene
- Alle generischen Video-Einstellungen: Laenge, Aufloesung, FPS, Seed, 
  Steps, Lightning-Modus an/aus

## Audio-Funktionalitaet

- Recherchiere, ob und wie Audio-Sync (Lip-Sync, Bewegungsrhythmus) in 
  den offenen Wan-2.2-Gewichten/Diffusers-Integration verfuegbar ist 
  (z.B. ueber Wan2.2-S2V, falls dies bereits offen mit Checkpoints 
  existiert)
- UI: optionales Audio-Upload-Feld (WAV/MP3) zur Sync-Steuerung, sofern 
  technisch verfuegbar
- Optional spaeter: separates lokales TTS-Modul als Vorstufe einplanen

## UI-Anforderungen

- Modell-Dropdown (Video/Bild getrennt)
- LoRA-Verwaltung: Mehrfachauswahl mit Staerke-Reglern, speicherbare 
  Presets (siehe oben, generisches System)
- Upload-Feld fuer Startbild
- Optionales Audio-Upload-Feld
- Prompt-Textfeld + negativer Prompt
- Video-Einstellungen: Laenge, Aufloesung, FPS, Seed, Steps, 
  Lightning-Modus an/aus
- Bild-Einstellungen: Aufloesung, Seed, Steps, CFG-Scale
- Fortschrittsanzeige, automatische VRAM-Erkennung mit Warnung

## Storage-Struktur

```
vid4me/
├── models/
│   ├── registry.json
│   ├── video/wan22-14b/
│   ├── image/chroma/
│   └── loras/
│       ├── video/
│       └── image/
├── outputs/<projekt>/
├── config/settings.json
├── backend/          # Python FastAPI, Modell-Loader, Inferenz
│   ├── core/          # Modell-Loader-Interface (abstrakt)
│   └── plugins/        # ein Loader-Modul pro Modell-Familie
└── frontend/          # Electron + React/Vue UI
```

## Erster Schritt - bevor Code geschrieben wird

1. Konkreter Aufbau des Modell-Loader-Interfaces fuer Plugin-
   Erweiterbarkeit, inkl. generischem LoRA-Stacking-Mechanismus
2. Diffusers-Integration fuer Wan 2.2 (14B MoE-Variante) und Chroma 
   pruefen (Pipeline-Klassen, Versionen, LoRA-Kompatibilitaet, 
   Lightning-Distillation-Support)
3. Audio-Sync-Recherche wie oben beschrieben durchfuehren
4. Electron+FastAPI-Grundgeruest: wie wird der Python-Prozess vom 
   Electron-Main-Process gestartet/verwaltet, wie erfolgt die 
   Kommunikation (REST/WebSocket)?
5. Projektstruktur bestaetigen/anpassen, Python-Paketliste 
   (diffusers, torch, CUDA-Version fuer RTX 4090) sowie Node/Electron-
   Abhaengigkeiten

## Weiteres Vorgehen

Danach: MVP mit Chroma fuer Bild-Generierung (Backend + einfaches 
Electron-Fenster), dann Wan 2.2 14B Basis-Video-Pipeline mit generischem 
LoRA-Support, dann Audio-Sync-Recherche einordnen, dann UI-Feinschliff 
und Modell-Registry final ausbauen.

WICHTIGE VRAM/RAM-KONFIGURATION für Wan 2.2 14B auf meiner Hardware 
(RTX 4090, 24GB VRAM, 64GB System-RAM):

- Wan 2.2 14B in FP16 braucht ca. 55-65GB VRAM - das passt NICHT auf 
  meine 4090
- Lösung: FP8-Quantisierung für den Diffusion-Transformer (bringt 
  Bedarf auf ca. 22-26GB) KOMBINIERT MIT CPU-Offloading des T5-XXL 
  Text-Encoders (via enable_model_cpu_offload() in Diffusers) - das 
  senkt den VRAM-Bedarf auf ca. 14-16GB
- Da ich 64GB System-RAM habe (deutlich über der 32GB-Empfehlung), 
  soll das Text-Encoder-CPU-Offloading als FESTER STANDARD implementiert 
  werden, nicht nur als optionaler Schalter
- Ziel: FP8-Transformer auf GPU + Text-Encoder-Offload auf RAM als 
  Default-Pipeline-Konfiguration beim Modell-Laden
- Berücksichtige das bereits im Modell-Loader-Interface/Plugin für 
  Wan 2.2, damit ich nicht nachträglich VRAM-Out-of-Memory-Fehler 
  debuggen muss
