import React, { useRef, useState } from "react";
import { api } from "../api/client.js";

/**
 * Generisches LoRA-Stacking-Panel.
 * - listet alle .safetensors im LoRA-Ordner des Modelltyps
 * - Mehrfachauswahl mit Staerke-Regler pro LoRA
 * - fuer Video zusaetzlich Experten-Ziel (beide/high/low)
 * - speicherbare, frei benennbare Presets
 */
export default function LoraPanel({ modelType, loras, stack, setStack, presets,
                                    onPresetsChanged, onLorasChanged }) {
  const fileInput = useRef(null);
  const [presetName, setPresetName] = useState("");
  const entries = Object.entries(loras ?? {}).filter(([, meta]) => !meta.missing);

  const stackFor = (file) => stack.find((s) => s.file === file);

  // Wan-2.2-Paar-LoRAs am Dateinamen erkennen ("HighNoise", "low_noise",
  // "_high_" usw.) und das Experten-Ziel passend vorbelegen
  function guessTarget(file) {
    const n = file.toLowerCase();
    if (/high[\s._-]?noise|[._-]high(?=[._-])/.test(n)) return "high";
    if (/low[\s._-]?noise|[._-]low(?=[._-])/.test(n)) return "low";
    return "both";
  }

  function toggle(file, meta) {
    if (stackFor(file)) {
      setStack(stack.filter((s) => s.file !== file));
    } else {
      setStack([...stack, {
        file,
        strength: meta.strength_default ?? 1.0,
        target: meta.target ?? guessTarget(file),
      }]);
    }
  }

  function update(file, patch) {
    setStack(stack.map((s) => (s.file === file ? { ...s, ...patch } : s)));
  }

  async function uploadLora(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    await api.uploadLora(modelType, file);
    onLorasChanged();
    e.target.value = "";
  }

  async function savePreset() {
    const name = presetName.trim();
    if (!name || stack.length === 0) return;
    await api.savePreset(modelType, name, stack);
    setPresetName("");
    onPresetsChanged();
  }

  function applyPreset(name) {
    const preset = presets?.[name];
    if (preset) setStack(preset.map((p) => ({ target: "both", ...p })));
  }

  async function removePreset(name) {
    await api.deletePreset(modelType, name);
    onPresetsChanged();
  }

  return (
    <div className="panel lora-panel">
      <div className="panel-header">
        <h3>LoRAs ({modelType === "video" ? "Video" : "Bild"})</h3>
        <div className="panel-actions">
          <button className="btn small" onClick={() => onLorasChanged()}
                  title="Ordner neu scannen">⟳ Scan</button>
          <button className="btn small" onClick={() => fileInput.current?.click()}>
            + Hochladen
          </button>
          <input ref={fileInput} type="file" accept=".safetensors"
                 style={{ display: "none" }} onChange={uploadLora} />
        </div>
      </div>

      {entries.length === 0 && (
        <p className="muted">
          Keine LoRAs gefunden. Dateien (.safetensors) in
          <code> models/loras/{modelType}/</code> ablegen oder hochladen.
        </p>
      )}

      <ul className="lora-list">
        {entries.map(([file, meta]) => {
          const active = stackFor(file);
          return (
            <li key={file} className={active ? "lora active" : "lora"}>
              <label className="lora-head">
                <input type="checkbox" checked={!!active}
                       onChange={() => toggle(file, meta)} />
                <span className="lora-name" title={file}>{meta.name || file}</span>
                <button
                  className="link danger lora-delete" title="LoRA-Datei loeschen"
                  onClick={async (e) => {
                    e.preventDefault();
                    if (!window.confirm(`LoRA-Datei „${file}" endgueltig loeschen?`)) return;
                    await api.deleteLora(modelType, file);
                    setStack(stack.filter((s) => s.file !== file));
                    onLorasChanged();
                  }}
                >🗑</button>
              </label>
              {active && (
                <div className="lora-controls">
                  <input
                    type="range" min="0" max="1.5" step="0.05"
                    value={active.strength}
                    onChange={(e) => update(file, { strength: Number(e.target.value) })}
                  />
                  <input
                    type="number" min="0" max="2" step="0.05" className="strength"
                    value={active.strength}
                    onChange={(e) => update(file, { strength: Number(e.target.value) })}
                  />
                  {modelType === "video" && (
                    <select
                      value={active.target}
                      title="Wan 2.2 MoE: auf welchen Experten anwenden"
                      onChange={(e) => update(file, { target: e.target.value })}
                    >
                      <option value="both">beide Experten</option>
                      <option value="high">nur High-Noise</option>
                      <option value="low">nur Low-Noise</option>
                    </select>
                  )}
                </div>
              )}
            </li>
          );
        })}
      </ul>

      <div className="preset-row">
        <input
          type="text" placeholder="Preset-Name…"
          value={presetName} onChange={(e) => setPresetName(e.target.value)}
        />
        <button className="btn small" disabled={!presetName.trim() || stack.length === 0}
                onClick={savePreset}>Speichern</button>
      </div>
      {Object.keys(presets ?? {}).length > 0 && (
        <ul className="preset-list">
          {Object.keys(presets).map((name) => (
            <li key={name}>
              <button className="link" onClick={() => applyPreset(name)}>{name}</button>
              <button className="link danger" title="Preset loeschen"
                      onClick={() => removePreset(name)}>✕</button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
