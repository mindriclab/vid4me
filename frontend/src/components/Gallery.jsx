import React from "react";
import { api } from "../api/client.js";

export default function Gallery({ outputs, onChanged }) {
  if (!outputs?.length) {
    return <p className="muted center">Noch keine Ergebnisse — Ausgaben landen in <code>outputs/</code>.</p>;
  }

  async function hide(f) {
    await api.hideOutput(f.path);
    onChanged?.();
  }

  return (
    <div className="gallery">
      {outputs.map((f) => (
        <figure key={f.path} className="gallery-item">
          <button
            className="gallery-hide" title="Aus Galerie entfernen (Datei bleibt in outputs/)"
            onClick={() => hide(f)}
          >✕</button>
          {f.kind === "video" ? (
            <video src={api.fileUrl(f.path)} controls loop muted />
          ) : (
            <img src={api.fileUrl(f.path)} alt={f.name} loading="lazy" />
          )}
          <figcaption title={f.path}>{f.name}</figcaption>
        </figure>
      ))}
    </div>
  );
}
