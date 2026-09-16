# Vid4me — AUF EIS GELEGT (Stand 2026-07-20)

**Status: pausiert / eingefroren.** Kein aktives Weiterentwickeln mehr.

## Warum
Vid4me (Electron + FastAPI + diffusers) sollte ein aufgeräumtes UI für lokale
Bild-/Video-Generierung sein. Ergebnis der Analyse am 2026-07-20:

- Für die tatsächliche Nutzung (lokal NSFW, Bild + Video auf einer RTX 4090)
  ist **ComfyUI überlegen**: läuft die vorhandenen Modelle schneller
  (fp8 + SageAttention statt bf16 + Offload) und aktueller.
- Vid4me **hinkt dem Ökosystem hinterher**: neue Modelle/Finetunes erscheinen
  zuerst in ComfyUI-Formaten, oft Custom-fp8, das diffusers gar nicht laden
  kann (z. B. der 10Eros-fp8-Checkpoint).
- Das Ziel „besseres, intuitiveres UI" wird stattdessen im Nachfolgeprojekt
  verfolgt: eine schlanke, schöne **Oberfläche VOR ComfyUI** (nutzt ComfyUIs
  API + die bestehenden Workflows, ohne Modelle zu duplizieren).

## Was fertig ist (falls je reaktiviert)
Grundgerüst komplett und getestet: Wan-2.2- und Chroma-Plugins, Plugin-System,
Modell-Download-Manager, Einzeldatei-Import (Wan), LoRA-Stacking, portable exe.
Zusätzlich am 2026-07-20 gebaut: **LTX-2.3-Plugin** (backend/plugins/ltx2.py)
für die offiziellen diffusers-LTX-2.3-Gewichte (T2V/I2V + Audio) — funktioniert,
aber ungetestet mit echten Gewichten; der 10Eros-Einzeldatei-Weg wurde bewusst
nicht fertiggebaut (lohnt nicht, siehe oben).

## Nachfolgeprojekt
Siehe D:\mindricS\My_Apps\ComfyDeck (schlankes ComfyUI-Frontend).
