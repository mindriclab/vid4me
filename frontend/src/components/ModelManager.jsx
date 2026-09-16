import React, { useEffect, useState } from "react";
import { api } from "../api/client.js";

/**
 * Modell-Manager (Modal):
 * - Registry-Modelle mit Download-Status, Download mit Live-Fortschritt
 * - Gewichte wieder loeschen (Speicher freigeben)
 * - lokal vorhandene Modelle einbinden (Ordner-Scan in models/ oder
 *   beliebiger Pfad, Diffusers-Format)
 * - Hugging-Face-Suche, um Modelle zur Registry hinzuzufuegen
 */
export default function ModelManager({ models, downloads, plugins, onClose, onModelsChanged }) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState(null);
  const [searching, setSearching] = useState(false);
  const [addForm, setAddForm] = useState(null);       // HF: {repo_id, type, plugin, task}
  const [importForm, setImportForm] = useState(null); // lokal: {path, type, plugin, task, name}
  const [localCandidates, setLocalCandidates] = useState([]);
  const [manualPath, setManualPath] = useState("");
  const [error, setError] = useState(null);
  const [notice, setNotice] = useState(null);

  const grouped = {
    video: models.filter((m) => m.type === "video"),
    image: models.filter((m) => m.type === "image"),
    components: models.filter((m) => m.type === "components"),
  };
  const [singleForm, setSingleForm] = useState(null); // {high_path, low_path, task, name}

  useEffect(() => {
    api.localScan().then(setLocalCandidates).catch(() => {});
  }, [models]);

  async function search() {
    if (!query.trim()) return;
    setSearching(true);
    setError(null);
    try {
      setResults(await api.hubSearch(query.trim()));
    } catch (err) {
      setError(err.message);
      setResults(null);
    } finally {
      setSearching(false);
    }
  }

  async function startDownload(id) {
    setError(null);
    try {
      await api.startDownload(id);
    } catch (err) {
      setError(err.message);
    }
  }

  async function removeModel(m) {
    if (!window.confirm(`„${m.name}" aus der Liste entfernen?\n(Dateien bleiben auf der Festplatte.)`)) return;
    await api.deleteModel(m.id);
    onModelsChanged();
  }

  async function removeModelFiles(m) {
    const size = m.size_gb ? ` (~${m.size_gb} GB werden frei)` : "";
    if (!window.confirm(`Heruntergeladene Gewichte von „${m.name}" wirklich loeschen${size}?\n\nDas Modell bleibt in der Liste und kann jederzeit neu geladen werden.`)) return;
    setError(null);
    try {
      const res = await api.deleteModelFiles(m.id);
      setNotice(`Gewichte geloescht — ${res.freed_gb} GB freigegeben.`);
      onModelsChanged();
    } catch (err) {
      setError(err.message);
    }
  }

  async function submitAdd() {
    setError(null);
    try {
      await api.addModel(addForm);
      setAddForm(null);
      onModelsChanged();
    } catch (err) {
      setError(err.message);
    }
  }

  async function submitImport() {
    setError(null);
    try {
      await api.importModel(importForm);
      setImportForm(null);
      setManualPath("");
      setNotice("Modell eingebunden — es steht jetzt im Dropdown zur Auswahl.");
      onModelsChanged();
    } catch (err) {
      setError(err.message);
    }
  }

  async function submitSingle() {
    setError(null);
    try {
      await api.importSingleFile({
        high_path: singleForm.high_path,
        low_path: singleForm.low_path || null,
        task: singleForm.task,
        name: singleForm.name || null,
      });
      setSingleForm(null);
      setNotice("Einzeldatei-Modell eingebunden. Fuer die Nutzung werden die "
        + "Wan 2.2 Basis-Komponenten benoetigt (siehe oben, einmalig ~11 GB).");
      onModelsChanged();
    } catch (err) {
      setError(err.message);
    }
  }

  function openImportForm(candidate) {
    setImportForm({
      path: candidate.path,
      type: candidate.type ?? "video",
      plugin: candidate.type === "image" ? "chroma" : "wan22",
      task: candidate.type === "image" ? "t2i" : "i2v",
      name: candidate.name ?? "",
      format: candidate.format,
    });
  }

  function ModelRow({ m }) {
    const dl = downloads[m.id];
    const active = dl && ["starting", "running", "cancelling"].includes(dl.status);
    const pct = Math.round((dl?.progress ?? 0) * 100);
    return (
      <li className="model-row">
        <div className="model-info">
          <strong>{m.name}{m.imported ? " (lokal)" : ""}</strong>
          <span className="muted small-text">
            {m.repo_id ?? m.local_path}
            {m.size_gb ? ` · ~${m.size_gb} GB` : ""}
            {m.format === "single_file" ? " · Einzeldatei" : ""}
          </span>
          {m.format === "single_file" && m.downloaded && !m.components_ready && (
            <span className="error-text small-text">
              ⚠ Basis-Komponenten fehlen — oben herunterladen (~11 GB)
            </span>
          )}
          {dl?.status === "error" && <span className="error-text small-text">{dl.error}</span>}
        </div>
        <div className="model-actions">
          {m.downloaded ? (
            <>
              <span className="chip ok">✓ vorhanden</span>
              {!m.imported && (
                <button className="btn small danger" title="Gewichte loeschen (Speicher freigeben)"
                        onClick={() => removeModelFiles(m)}>🗑 Dateien</button>
              )}
            </>
          ) : active ? (
            <>
              <div className="progress-bar mini">
                <div className="progress-fill" style={{ width: `${pct}%` }} />
                <span className="progress-text">
                  {dl.total_gb
                    ? `${dl.downloaded_gb ?? 0} / ${dl.total_gb} GB`
                    : `${dl.downloaded_gb ?? 0} GB`}
                </span>
              </div>
              <button className="btn small danger" onClick={() => api.cancelDownload(m.id)}>✕</button>
            </>
          ) : m.repo_id ? (
            <button className="btn small primary" onClick={() => startDownload(m.id)}>
              ⬇ Herunterladen
            </button>
          ) : (
            <span className="chip">Dateien fehlen</span>
          )}
          {m.custom && !active && (
            <button className="btn small danger" title="Aus Registry entfernen"
                    onClick={() => removeModel(m)}>✕ Eintrag</button>
          )}
        </div>
      </li>
    );
  }

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="panel-header">
          <h3>Modelle verwalten</h3>
          <button className="btn small" onClick={onClose}>Schliessen ✕</button>
        </div>

        {error && <div className="banner error">✕ {error}</div>}
        {notice && <div className="banner ok">✓ {notice}</div>}

        <h4>Videomodelle</h4>
        <ul className="model-list">
          {grouped.video.map((m) => <ModelRow key={m.id} m={m} />)}
        </ul>
        <h4>Bildmodelle</h4>
        <ul className="model-list">
          {grouped.image.map((m) => <ModelRow key={m.id} m={m} />)}
        </ul>

        {grouped.components.length > 0 && (
          <>
            <h4>Basis-Komponenten (für Einzeldatei-Modelle)</h4>
            <p className="muted small-text">
              Einzeldatei-Checkpoints (ComfyUI-Format) enthalten nur den Transformer —
              VAE, Text-Encoder &amp; Co. kommen aus diesem Paket (einmalig laden).
            </p>
            <ul className="model-list">
              {grouped.components.map((m) => <ModelRow key={m.id} m={m} />)}
            </ul>
          </>
        )}

        <h4>Lokales Modell einbinden</h4>
        <p className="muted small-text">
          Bereits vorhandene Modelle in <code>models/video/</code> bzw. <code>models/image/</code>{" "}
          kopieren — sie erscheinen dann hier. Unterstuetzt: Diffusers-Ordner und{" "}
          <strong>Einzeldatei-Checkpoints (.safetensors, ComfyUI-Format)</strong>. Alternativ
          direkt einen Ordnerpfad angeben (Modell bleibt an Ort und Stelle).
        </p>
        {localCandidates.length > 0 && (
          <ul className="model-list">
            {localCandidates.map((c) => (
              <li key={c.path} className="model-row">
                <div className="model-info">
                  <strong>{c.name}</strong>
                  <span className="muted small-text">
                    {c.path} · {c.size_gb} GB ·{" "}
                    {c.format === "diffusers" ? "Diffusers-Format" : "Einzeldatei (ComfyUI-Format)"}
                  </span>
                </div>
                <button className="btn small primary"
                        onClick={() => c.format === "diffusers"
                          ? openImportForm(c)
                          : setSingleForm({ high_path: c.path, low_path: "", task: "i2v", name: c.name })}>
                  + Einbinden
                </button>
              </li>
            ))}
          </ul>
        )}
        <div className="preset-row">
          <input
            type="text" placeholder="Oder Pfad zu einem Modell-Ordner (z.B. D:\Modelle\wan22)…"
            value={manualPath}
            onChange={(e) => setManualPath(e.target.value)}
          />
          <button className="btn small" disabled={!manualPath.trim()}
                  onClick={() => openImportForm({ path: manualPath.trim(), name: "", format: "diffusers" })}>
            Pruefen
          </button>
        </div>

        {singleForm && (
          <div className="add-form">
            <strong>Einzeldatei einbinden: {singleForm.high_path}</strong>
            <div className="grid3">
              <label>Aufgabe
                <select value={singleForm.task}
                        onChange={(e) => setSingleForm({ ...singleForm, task: e.target.value })}>
                  <option value="i2v">Bild → Video</option>
                  <option value="t2v">Text → Video</option>
                </select>
              </label>
              <label>Low-Noise-Datei (optional)
                <select value={singleForm.low_path}
                        onChange={(e) => setSingleForm({ ...singleForm, low_path: e.target.value })}>
                  <option value="">— keine (merged Checkpoint) —</option>
                  {localCandidates
                    .filter((c) => c.format === "single_file" && c.path !== singleForm.high_path)
                    .map((c) => <option key={c.path} value={c.path}>{c.name}</option>)}
                </select>
              </label>
              <label>Anzeigename
                <input type="text" value={singleForm.name}
                       onChange={(e) => setSingleForm({ ...singleForm, name: e.target.value })} />
              </label>
            </div>
            <p className="muted small-text">
              Wan 2.2 A14B kommt als <strong>Paar</strong> (High- + Low-Noise-Datei) — dann beide
              angeben. Merged/kombinierte Checkpoints (eine Datei) laufen mit einem Transformer.
              Zusaetzlich einmalig die Basis-Komponenten (~11 GB) herunterladen.
            </p>
            <div className="preset-row">
              <button className="btn small primary" onClick={submitSingle}>Einbinden</button>
              <button className="btn small" onClick={() => setSingleForm(null)}>Abbrechen</button>
            </div>
          </div>
        )}

        {importForm && (
          <div className="add-form">
            <strong>Einbinden: {importForm.path}</strong>
            <div className="grid3">
              <label>Typ
                <select value={importForm.type}
                        onChange={(e) => setImportForm({ ...importForm, type: e.target.value,
                          plugin: e.target.value === "image" ? "chroma" : "wan22",
                          task: e.target.value === "image" ? "t2i" : "i2v" })}>
                  <option value="video">Video</option>
                  <option value="image">Bild</option>
                </select>
              </label>
              <label>Plugin
                <select value={importForm.plugin}
                        onChange={(e) => setImportForm({ ...importForm, plugin: e.target.value })}>
                  {plugins.map((p) => <option key={p} value={p}>{p}</option>)}
                </select>
              </label>
              <label>Aufgabe
                <select value={importForm.task}
                        onChange={(e) => setImportForm({ ...importForm, task: e.target.value })}>
                  {importForm.type === "video" ? (
                    <>
                      <option value="i2v">Bild → Video</option>
                      <option value="t2v">Text → Video</option>
                    </>
                  ) : (
                    <option value="t2i">Text → Bild</option>
                  )}
                </select>
              </label>
            </div>
            <div className="preset-row">
              <input type="text" placeholder="Anzeigename (optional)"
                     value={importForm.name}
                     onChange={(e) => setImportForm({ ...importForm, name: e.target.value })} />
            </div>
            <div className="preset-row" style={{ marginTop: 8 }}>
              <button className="btn small primary" onClick={submitImport}>Einbinden</button>
              <button className="btn small" onClick={() => setImportForm(null)}>Abbrechen</button>
            </div>
          </div>
        )}

        <h4>Hugging Face durchsuchen</h4>
        <div className="preset-row">
          <input
            type="text" placeholder="Modell suchen (z.B. wan lora, flux…)"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && search()}
          />
          <button className="btn small" disabled={searching} onClick={search}>
            {searching ? "Suche…" : "Suchen"}
          </button>
        </div>

        {results && (
          <ul className="model-list search-results">
            {results.length === 0 && <li className="muted">Keine Treffer.</li>}
            {results.map((r) => (
              <li key={r.repo_id} className="model-row">
                <div className="model-info">
                  <strong>{r.repo_id}</strong>
                  <span className="muted small-text">
                    {r.pipeline_tag ? `${r.pipeline_tag} · ` : ""}
                    {r.downloads != null ? `${r.downloads.toLocaleString("de-DE")} Downloads` : ""}
                  </span>
                </div>
                <button
                  className="btn small"
                  onClick={() => setAddForm({
                    repo_id: r.repo_id, type: "video", plugin: plugins[0] ?? "wan22", task: "t2v",
                  })}
                >
                  + Hinzufuegen
                </button>
              </li>
            ))}
          </ul>
        )}

        {addForm && (
          <div className="add-form">
            <strong>{addForm.repo_id}</strong>
            <div className="grid3">
              <label>Typ
                <select value={addForm.type}
                        onChange={(e) => setAddForm({ ...addForm, type: e.target.value,
                          task: e.target.value === "image" ? "t2i" : "t2v" })}>
                  <option value="video">Video</option>
                  <option value="image">Bild</option>
                </select>
              </label>
              <label>Plugin
                <select value={addForm.plugin}
                        onChange={(e) => setAddForm({ ...addForm, plugin: e.target.value })}>
                  {plugins.map((p) => <option key={p} value={p}>{p}</option>)}
                </select>
              </label>
              <label>Aufgabe
                <select value={addForm.task}
                        onChange={(e) => setAddForm({ ...addForm, task: e.target.value })}>
                  {addForm.type === "video" ? (
                    <>
                      <option value="t2v">Text → Video</option>
                      <option value="i2v">Bild → Video</option>
                    </>
                  ) : (
                    <option value="t2i">Text → Bild</option>
                  )}
                </select>
              </label>
            </div>
            <p className="muted small-text">
              Hinweis: Das gewaehlte Plugin muss zur Modell-Familie passen
              (wan22 fuer Wan-Videomodelle im Diffusers-Format, chroma fuer Chroma/Flux-Bildmodelle).
            </p>
            <div className="preset-row">
              <button className="btn small primary" onClick={submitAdd}>Zur Registry hinzufuegen</button>
              <button className="btn small" onClick={() => setAddForm(null)}>Abbrechen</button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
