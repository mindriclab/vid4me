import React from "react";
import { api } from "../api/client.js";

const STATUS_LABELS = {
  queued: "In Warteschlange",
  loading: "Modell laedt",
  running: "Generierung",
  done: "Fertig",
  error: "Fehler",
  cancelled: "Abgebrochen",
};

export default function ProgressPanel({ job }) {
  if (!job) return null;
  const pct = Math.round((job.progress ?? 0) * 100);
  const active = ["queued", "loading", "running"].includes(job.status);

  return (
    <div className={`panel progress-panel status-${job.status}`}>
      <div className="progress-head">
        <strong>{STATUS_LABELS[job.status] ?? job.status}</strong>
        <span className="muted">{job.message}</span>
        {active && (
          <button className="btn small danger" onClick={() => api.cancelJob(job.id)}>
            Abbrechen
          </button>
        )}
      </div>
      {active && (
        <div className="progress-bar">
          <div className="progress-fill" style={{ width: `${pct}%` }} />
          <span className="progress-text">
            {job.total_steps > 0 ? `${job.step}/${job.total_steps} · ${pct}%` : "…"}
          </span>
        </div>
      )}
      {job.status === "error" && <p className="error-text">{job.error}</p>}
      {job.status === "done" && job.seed != null && (
        <p className="muted">Seed: {job.seed}</p>
      )}
    </div>
  );
}
