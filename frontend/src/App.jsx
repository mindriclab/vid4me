import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, connectJobSocket } from "./api/client.js";
import LoraPanel from "./components/LoraPanel.jsx";
import ProgressPanel from "./components/ProgressPanel.jsx";
import Gallery from "./components/Gallery.jsx";
import ModelManager from "./components/ModelManager.jsx";
import {
  Field, NumberField, SelectField, Toggle,
  RESOLUTIONS_VIDEO, RESOLUTIONS_IMAGE,
} from "./components/Fields.jsx";

const DEFAULT_IMAGE_PARAMS = {
  modelId: "", prompt: "", negative: "",
  resolution: "1024x1024", steps: 40, guidance: 4.0, seed: -1,
};

// Offizieller Wan-Standard-Negativprompt (chinesisch, wie in den
// ComfyUI-Vorlagen — UMT5 ist mehrsprachig). Vom Nutzer frei aenderbar.
const WAN_DEFAULT_NEGATIVE =
  "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，" +
  "低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，" +
  "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走";

const DEFAULT_VIDEO_PARAMS = {
  modelId: "", prompt: "", negative: WAN_DEFAULT_NEGATIVE,
  resolution: "832x480", steps: 40, guidance: 3.5, seed: -1,
  lengthSeconds: 5, fps: 16, lightning: false,
  imagePath: null, imageName: "", audioPath: null, audioName: "",
};

export default function App() {
  const [tab, setTab] = useState("image");
  const [backendUp, setBackendUp] = useState(false);
  const [system, setSystem] = useState(null);
  const [models, setModels] = useState([]);
  const [loras, setLoras] = useState({ video: {}, image: {} });
  const [presets, setPresets] = useState({ video: {}, image: {} });
  const [imageParams, setImageParams] = useState(() => ({
    ...DEFAULT_IMAGE_PARAMS,
    modelId: localStorage.getItem("vid4me.modelId.image") || "",
  }));
  const [videoParams, setVideoParams] = useState(() => ({
    ...DEFAULT_VIDEO_PARAMS,
    modelId: localStorage.getItem("vid4me.modelId.video") || "",
  }));
  const [imageStack, setImageStack] = useState([]);
  const [videoStack, setVideoStack] = useState([]);
  const [job, setJob] = useState(null);
  const [outputs, setOutputs] = useState([]);
  const [vramWarning, setVramWarning] = useState(null);
  const [error, setError] = useState(null);
  const [showModels, setShowModels] = useState(false);
  const [downloads, setDownloads] = useState({});
  const startImageInput = useRef(null);
  const audioInput = useRef(null);

  const params = tab === "image" ? imageParams : videoParams;
  const setParams = tab === "image" ? setImageParams : setVideoParams;
  const stack = tab === "image" ? imageStack : videoStack;
  const setStack = tab === "image" ? setImageStack : setVideoStack;

  const modelsOfType = useMemo(
    () => models.filter((m) => m.type === tab),
    [models, tab]
  );
  const selectedModel = modelsOfType.find((m) => m.id === params.modelId);

  // ---- Daten laden ----------------------------------------------------------

  const refreshAll = useCallback(async () => {
    try {
      const [sys, mods, lor, presetsVideo, presetsImage, outs, dls] = await Promise.all([
        api.system(), api.models(), api.loras(),
        api.presets("video"), api.presets("image"), api.outputs(), api.downloads(),
      ]);
      setSystem(sys);
      setModels(mods);
      setLoras(lor);
      setPresets({ video: presetsVideo, image: presetsImage });
      setOutputs(outs);
      setDownloads(Object.fromEntries(dls.map((d) => [d.model_id, d])));
      setBackendUp(true);
    } catch {
      setBackendUp(false);
    }
  }, []);

  useEffect(() => {
    refreshAll();
    const poll = setInterval(async () => {
      try {
        // System + Modell-Liste mitpollen, damit die Anzeige nie veraltet
        // (z.B. Downloads/Importe, die ausserhalb dieses Fensters passieren)
        const [sys, mods] = await Promise.all([api.system(), api.models()]);
        setSystem(sys);
        setModels(mods);
        setBackendUp(true);
      } catch {
        setBackendUp(false);
      }
    }, 10000);
    return () => clearInterval(poll);
  }, [refreshAll]);

  // Beim ersten Backend-Kontakt nachladen
  useEffect(() => {
    if (backendUp && models.length === 0) refreshAll();
  }, [backendUp, models.length, refreshAll]);

  // Standard-Modell je Tab setzen: gespeicherte Auswahl behalten, sonst
  // bevorzugt ein bereits heruntergeladenes Modell (nie still eins waehlen,
  // dessen Gewichte fehlen)
  useEffect(() => {
    if (modelsOfType.length === 0) return;
    if (!modelsOfType.some((m) => m.id === params.modelId)) {
      const pick = modelsOfType.find((m) => m.downloaded) || modelsOfType[0];
      setParams((p) => ({ ...p, modelId: pick.id }));
    }
  }, [modelsOfType, params.modelId, setParams]);

  // Modellauswahl ueber App-Neustarts merken
  useEffect(() => {
    if (imageParams.modelId)
      localStorage.setItem("vid4me.modelId.image", imageParams.modelId);
  }, [imageParams.modelId]);
  useEffect(() => {
    if (videoParams.modelId)
      localStorage.setItem("vid4me.modelId.video", videoParams.modelId);
  }, [videoParams.modelId]);

  // WebSocket: Job- und Download-Updates
  useEffect(() => {
    const disconnect = connectJobSocket((event) => {
      if (event.type === "job") {
        setJob(event.job);
        if (event.job.status === "done") {
          api.outputs().then(setOutputs).catch(() => {});
        }
      }
      if (event.type === "download") {
        const dl = event.download;
        setDownloads((prev) => ({ ...prev, [dl.model_id]: dl }));
        if (dl.status === "done") {
          api.models().then(setModels).catch(() => {});
        }
      }
    });
    return disconnect;
  }, []);

  // VRAM-Check bei Modellwechsel
  useEffect(() => {
    if (!params.modelId || !backendUp) return;
    api.vramCheck(params.modelId)
      .then((res) => setVramWarning(res.warning))
      .catch(() => setVramWarning(null));
  }, [params.modelId, backendUp]);

  // ---- Aktionen ----------------------------------------------------------------

  async function uploadStartImage(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    const { path } = await api.upload(file);
    setVideoParams((p) => ({ ...p, imagePath: path, imageName: file.name }));
    e.target.value = "";
  }

  async function uploadAudio(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    const { path } = await api.upload(file);
    setVideoParams((p) => ({ ...p, audioPath: path, audioName: file.name }));
    e.target.value = "";
  }

  async function generate() {
    setError(null);
    const [width, height] = params.resolution.split("x").map(Number);
    const body = {
      model_id: params.modelId,
      prompt: params.prompt,
      negative_prompt: params.negative,
      seed: params.seed === "" ? -1 : params.seed,
      steps: params.lightning ? Math.min(params.steps, 8) : params.steps,
      guidance_scale: params.guidance,
      width, height,
      loras: stack.map(({ file, strength, target }) => ({ file, strength, target })),
      project: "default",
    };
    if (tab === "video") {
      body.num_frames = Math.round(params.lengthSeconds * params.fps / 4) * 4 + 1;
      body.fps = params.fps;
      body.lightning = params.lightning;
      body.image_path = params.imagePath;
      body.audio_path = params.audioPath;
    }
    try {
      const j = await api.generate(body);
      setJob(j);
    } catch (err) {
      setError(err.message);
    }
  }

  const busy = job && ["queued", "loading", "running"].includes(job.status);
  const isI2V = selectedModel?.task === "i2v";

  // ---- Render ---------------------------------------------------------------------

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="logo">▶</span> Vid4me
          <span className="tagline">lokale Bild- &amp; Videogenerierung</span>
        </div>
        <div className="topbar-right">
          <button className="btn small" onClick={() => setShowModels(true)}>
            ⬇ Modelle
          </button>
          <VramBadge system={system} backendUp={backendUp} />
        </div>
      </header>

      {!backendUp && (
        <div className="banner warn">
          Backend nicht erreichbar — es wird automatisch gestartet, einen Moment…
        </div>
      )}
      {backendUp && system && !system.torch_available && (
        <div className="banner warn">
          PyTorch/Diffusers sind noch nicht installiert. Bitte einmalig{" "}
          <code>scripts\setup_backend.ps1</code> ausfuehren (siehe README).
        </div>
      )}
      {vramWarning && <div className="banner warn">⚠ {vramWarning}</div>}
      {error && <div className="banner error">✕ {error}</div>}

      <nav className="tabs">
        <button className={tab === "image" ? "tab active" : "tab"}
                onClick={() => setTab("image")}>Bild</button>
        <button className={tab === "video" ? "tab active" : "tab"}
                onClick={() => setTab("video")}>Video</button>
      </nav>

      <main className="layout">
        <section className="column controls">
          <div className="panel">
            <SelectField
              label={tab === "image" ? "Bildmodell" : "Videomodell"}
              value={params.modelId}
              onChange={(v) => setParams((p) => ({ ...p, modelId: v }))}
              options={modelsOfType.map((m) => ({
                value: m.id,
                label: m.downloaded ? m.name : `${m.name} — nicht heruntergeladen`,
              }))}
            />
            {selectedModel && !selectedModel.downloaded && (
              <p className="muted small-text">
                Gewichte fehlen lokal —{" "}
                <button className="link" onClick={() => setShowModels(true)}>
                  jetzt im Modell-Manager herunterladen
                </button>
              </p>
            )}
            {selectedModel?.format === "single_file" && !selectedModel.components_ready && (
              <p className="muted small-text">
                ⚠ Einzeldatei-Modell:{" "}
                <button className="link" onClick={() => setShowModels(true)}>
                  Wan 2.2 Basis-Komponenten herunterladen
                </button>{" "}
                (einmalig ~11 GB), sonst kann nicht generiert werden.
              </p>
            )}

            <Field label="Prompt">
              <textarea rows={4} value={params.prompt} placeholder="Was soll entstehen?"
                        onChange={(e) => setParams((p) => ({ ...p, prompt: e.target.value }))} />
            </Field>
            <Field label="Negativer Prompt">
              <textarea rows={2} value={params.negative} placeholder="Was vermieden werden soll…"
                        onChange={(e) => setParams((p) => ({ ...p, negative: e.target.value }))} />
            </Field>

            {tab === "video" && (
              <>
                <Field label={isI2V ? "Startbild (erforderlich)" : "Startbild (I2V-Modell waehlen)"}>
                  <div className="upload-row">
                    <button className="btn" onClick={() => startImageInput.current?.click()}>
                      {videoParams.imageName || "Bild auswaehlen…"}
                    </button>
                    {videoParams.imagePath && (
                      <button className="btn small danger" title="Entfernen"
                              onClick={(e) => {
                                e.preventDefault();
                                setVideoParams((p) => ({ ...p, imagePath: null, imageName: "" }));
                              }}>
                        ✕
                      </button>
                    )}
                  </div>
                  {videoParams.imagePath && (
                    <img
                      className="start-image-preview"
                      src={api.fileUrl(videoParams.imagePath)}
                      alt="Startbild-Vorschau"
                    />
                  )}
                </Field>
                {/* Datei-Inputs BEWUSST ausserhalb des Labels: ein Klick irgendwo
                    im Label (z.B. aufs ✕) wuerde sonst den Datei-Dialog oeffnen */}
                <input ref={startImageInput} type="file" accept=".png,.jpg,.jpeg,.webp"
                       style={{ display: "none" }} onChange={uploadStartImage} />
                <Field label="Audio (Sync)"
                       hint="Wan2.2-S2V ist noch nicht in Diffusers integriert — Feld ist fuer ein kommendes Plugin reserviert.">
                  <div className="upload-row">
                    <button className="btn" disabled title="Noch nicht verfuegbar (S2V-Diffusers-Integration ausstehend)">
                      {videoParams.audioName || "Noch nicht verfuegbar"}
                    </button>
                  </div>
                </Field>
                <input ref={audioInput} type="file" accept=".wav,.mp3"
                       style={{ display: "none" }} onChange={uploadAudio} />
              </>
            )}
          </div>

          <div className="panel">
            <h3>Einstellungen</h3>
            <div className="grid2">
              <SelectField label="Aufloesung" value={params.resolution}
                onChange={(v) => setParams((p) => ({ ...p, resolution: v }))}
                options={tab === "video" ? RESOLUTIONS_VIDEO : RESOLUTIONS_IMAGE} />
              <NumberField label="Seed (-1 = zufaellig)" value={params.seed}
                onChange={(v) => setParams((p) => ({ ...p, seed: v }))} min={-1} />
              <NumberField label="Steps" value={params.steps} min={1} max={100}
                onChange={(v) => setParams((p) => ({ ...p, steps: v }))} />
              <NumberField label="CFG-Scale" value={params.guidance} min={0} max={20} step={0.5}
                onChange={(v) => setParams((p) => ({ ...p, guidance: v }))} />
              {tab === "video" && (
                <>
                  <NumberField label="Laenge (Sekunden)" value={videoParams.lengthSeconds}
                    min={1} max={10} onChange={(v) => setVideoParams((p) => ({ ...p, lengthSeconds: v }))} />
                  <SelectField label="FPS" value={String(videoParams.fps)}
                    onChange={(v) => setVideoParams((p) => ({ ...p, fps: Number(v) }))}
                    options={[{ value: "16", label: "16" }, { value: "24", label: "24" }]} />
                </>
              )}
            </div>
            {tab === "video" && (
              <Toggle
                label="Lightning-Modus (4–8 Schritte, distilliert)"
                checked={videoParams.lightning}
                onChange={(v) => setVideoParams((p) => ({ ...p, lightning: v }))}
                hint="Benoetigt passende Lightning-/Self-Forcing-LoRAs im Stack (High- und Low-Noise-Datei). CFG wird automatisch auf 1.0 gesetzt."
              />
            )}
          </div>

          <button
            className="btn primary generate"
            disabled={busy || !params.modelId || !params.prompt.trim() ||
                      !selectedModel?.downloaded ||
                      (selectedModel?.format === "single_file" &&
                        !selectedModel.components_ready) ||
                      (isI2V && !videoParams.imagePath)}
            onClick={generate}
          >
            {busy ? "Läuft…" : tab === "image" ? "Bild generieren" : "Video generieren"}
          </button>

          <ProgressPanel job={job} />
        </section>

        <section className="column side">
          <LoraPanel
            modelType={tab}
            loras={loras[tab]}
            stack={stack}
            setStack={setStack}
            presets={presets[tab]}
            onPresetsChanged={() =>
              api.presets(tab).then((p) => setPresets((all) => ({ ...all, [tab]: p })))}
            onLorasChanged={() => api.loras().then(setLoras)}
          />
        </section>

        <section className="column results">
          <div className="panel grow">
            <div className="panel-header">
              <h3>Ergebnisse</h3>
              <div className="panel-actions">
                {window.vid4me?.openDataFolder && (
                  <button className="btn small" title="Output-Ordner im Explorer oeffnen"
                          onClick={() => window.vid4me.openDataFolder("outputs")}>
                    📂 Ordner
                  </button>
                )}
                <button className="btn small" onClick={() => api.outputs().then(setOutputs)}>⟳</button>
              </div>
            </div>
            <Gallery outputs={outputs}
                     onChanged={() => api.outputs().then(setOutputs)} />
          </div>
        </section>
      </main>

      {showModels && (
        <ModelManager
          models={models}
          downloads={downloads}
          plugins={system?.plugins ?? []}
          onClose={() => setShowModels(false)}
          onModelsChanged={() => api.models().then(setModels)}
        />
      )}
    </div>
  );
}

function VramBadge({ system, backendUp }) {
  if (!backendUp) return <span className="vram-badge offline">Backend offline</span>;
  const vram = system?.vram;
  if (!vram?.available) return <span className="vram-badge warn">Keine GPU</span>;
  const pct = vram.total_gb ? Math.round((vram.used_gb / vram.total_gb) * 100) : 0;
  return (
    <span className={`vram-badge ${pct > 85 ? "warn" : ""}`}
          title={`${vram.device_name} — ${vram.used_gb} / ${vram.total_gb} GB belegt`}>
      {vram.device_name?.replace("NVIDIA GeForce ", "")} · {vram.free_gb} GB frei
    </span>
  );
}
