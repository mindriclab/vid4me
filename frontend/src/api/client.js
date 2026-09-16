// API-Client: spricht mit dem lokalen FastAPI-Backend.
// In Electron liefert preload.cjs die Backend-URL; im Browser-Dev-Modus Fallback.
export const BASE = window.vid4me?.backendUrl ?? "http://127.0.0.1:8756";

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: options.body instanceof FormData
      ? undefined
      : { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail ?? detail; } catch { /* kein JSON */ }
    throw new Error(detail);
  }
  return res.json();
}

export const api = {
  system: () => request("/api/system"),
  vramCheck: (modelId) => request(`/api/system/vram-check/${modelId}`),
  unload: () => request("/api/system/unload", { method: "POST" }),
  models: (type) => request(`/api/models${type ? `?type=${type}` : ""}`),
  addModel: (body) => request("/api/models", { method: "POST", body: JSON.stringify(body) }),
  deleteModel: (id) => request(`/api/models/${encodeURIComponent(id)}`, { method: "DELETE" }),
  deleteModelFiles: (id) => request(`/api/models/${encodeURIComponent(id)}/files`, { method: "DELETE" }),
  localScan: () => request("/api/models/local-scan"),
  importModel: (body) => request("/api/models/import", { method: "POST", body: JSON.stringify(body) }),
  importSingleFile: (body) => request("/api/models/import-single", { method: "POST", body: JSON.stringify(body) }),
  deleteLora: (type, filename) =>
    request(`/api/loras/${type}/${encodeURIComponent(filename)}`, { method: "DELETE" }),
  downloads: () => request("/api/downloads"),
  startDownload: (id) => request(`/api/models/${encodeURIComponent(id)}/download`, { method: "POST" }),
  cancelDownload: (id) => request(`/api/models/${encodeURIComponent(id)}/download/cancel`, { method: "POST" }),
  hubSearch: (q) => request(`/api/hub/search?q=${encodeURIComponent(q)}`),
  loras: () => request("/api/loras"),
  updateLora: (type, filename, meta) =>
    request(`/api/loras/${type}/${encodeURIComponent(filename)}`, {
      method: "PATCH", body: JSON.stringify(meta),
    }),
  uploadLora: (type, file) => {
    const form = new FormData();
    form.append("file", file);
    return request(`/api/loras/${type}/upload`, { method: "POST", body: form });
  },
  presets: (type) => request(`/api/lora-presets/${type}`),
  savePreset: (type, name, stack) =>
    request(`/api/lora-presets/${type}`, {
      method: "PUT", body: JSON.stringify({ name, stack }),
    }),
  deletePreset: (type, name) =>
    request(`/api/lora-presets/${type}/${encodeURIComponent(name)}`, { method: "DELETE" }),
  upload: (file) => {
    const form = new FormData();
    form.append("file", file);
    return request("/api/upload", { method: "POST", body: form });
  },
  generate: (body) => request("/api/generate", { method: "POST", body: JSON.stringify(body) }),
  jobs: () => request("/api/jobs"),
  cancelJob: (id) => request(`/api/jobs/${id}/cancel`, { method: "POST" }),
  outputs: (project) => request(`/api/outputs${project ? `?project=${project}` : ""}`),
  hideOutput: (path) => request("/api/outputs/hide", { method: "POST", body: JSON.stringify({ path }) }),
  fileUrl: (path) => `${BASE}/api/file?path=${encodeURIComponent(path)}`,
};

// WebSocket fuer Job-Fortschritt; reconnect-freundlich.
export function connectJobSocket(onEvent) {
  const wsUrl = BASE.replace(/^http/, "ws") + "/ws";
  let ws = null;
  let closed = false;

  function connect() {
    ws = new WebSocket(wsUrl);
    ws.onmessage = (msg) => {
      try { onEvent(JSON.parse(msg.data)); } catch { /* ignorieren */ }
    };
    ws.onclose = () => {
      if (!closed) setTimeout(connect, 2000);
    };
  }
  connect();
  return () => { closed = true; ws?.close(); };
}
