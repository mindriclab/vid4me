/**
 * Electron-Main-Process.
 *
 * - startet das Python-FastAPI-Backend automatisch als Kindprozess und
 *   beendet es sauber beim Schliessen
 * - Dev-Modus: Projektordner = App- und Datenordner
 * - Gepackte App (Installer/Portable): App-Dateien liegen unter resources/,
 *   alle Daten (Modelle, Outputs, Config, Python-Umgebung) unter
 *   %USERPROFILE%\Vid4me — beim ersten Start wird die Python-Umgebung
 *   automatisch eingerichtet (scripts/setup_backend.ps1).
 */
const { app, BrowserWindow, dialog, ipcMain, Menu, shell } = require("electron");
const { spawn, execSync } = require("child_process");
const path = require("path");
const fs = require("fs");
const http = require("http");

const IS_PACKAGED = app.isPackaged;
// APP_ROOT: enthaelt backend/, scripts/ (dev: Projektordner, gepackt: resources/)
const APP_ROOT = IS_PACKAGED
  ? process.resourcesPath
  : path.resolve(__dirname, "..", "..");
// DATA_ROOT: Modelle, Outputs, Config, .venv
const DATA_ROOT = IS_PACKAGED
  ? path.join(app.getPath("home"), "Vid4me")
  : APP_ROOT;
// preload.cjs liest hierueber die Backend-Portkonfiguration
process.env.VID4ME_DATA_ROOT = DATA_ROOT;

let backendProc = null;
let mainWindow = null;

// ---- Datenordner-Bootstrap (nur gepackte App) --------------------------------

function ensureDataRoot() {
  const dirs = [
    "config", "outputs", "uploads",
    "models", "models/video", "models/image",
    "models/loras/video", "models/loras/image",
  ];
  for (const d of dirs) fs.mkdirSync(path.join(DATA_ROOT, d), { recursive: true });

  const templates = [
    ["templates/settings.json", "config/settings.json"],
    ["templates/registry.json", "models/registry.json"],
  ];
  for (const [from, to] of templates) {
    const src = path.join(APP_ROOT, from);
    const dst = path.join(DATA_ROOT, to);
    if (!fs.existsSync(dst) && fs.existsSync(src)) fs.copyFileSync(src, dst);
  }

  // Gepackte App nutzt Port 8757, damit sie nie mit einem laufenden
  // Dev-Backend (8756, Projektordner) kollidiert.
  try {
    const settingsPath = path.join(DATA_ROOT, "config", "settings.json");
    const settings = JSON.parse(fs.readFileSync(settingsPath, "utf-8"));
    if ((settings.backend?.port ?? 8756) === 8756) {
      settings.backend = { ...settings.backend, port: 8757 };
      fs.writeFileSync(settingsPath, JSON.stringify(settings, null, 2));
    }
  } catch { /* Settings unlesbar - Backend faellt auf Default zurueck */ }
}

function readBackendPort() {
  try {
    const settings = JSON.parse(
      fs.readFileSync(path.join(DATA_ROOT, "config", "settings.json"), "utf-8")
    );
    return settings.backend?.port ?? 8756;
  } catch {
    return 8756;
  }
}

// ---- Python / Erst-Setup ------------------------------------------------------

function venvPython() {
  return path.join(DATA_ROOT, ".venv", "Scripts", "python.exe");
}

function pythonExecutable() {
  const venv = venvPython();
  if (fs.existsSync(venv)) return venv;
  return "python"; // Dev-Fallback: System-Python
}

function sendSetupLog(line) {
  try {
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send("setup-log", line);
    }
  } catch { /* Fenster bereits zu */ }
}

function runFirstTimeSetup() {
  // Fortschritt wird live in die Einrichtungs-Seite (setup.html) gestreamt.
  return new Promise((resolve) => {
    const script = path.join(APP_ROOT, "scripts", "setup_backend.ps1");
    const proc = spawn(
      "powershell.exe",
      ["-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script, "-DataRoot", DATA_ROOT],
      { stdio: ["ignore", "pipe", "pipe"], windowsHide: true }
    );
    proc.stdout.on("data", (d) => sendSetupLog(d.toString()));
    proc.stderr.on("data", (d) => sendSetupLog(d.toString()));
    proc.on("exit", (code) => resolve(code === 0));
    proc.on("error", (err) => {
      sendSetupLog(`FEHLER: ${err.message}`);
      resolve(false);
    });
  });
}

// ---- Backend-Lebenszyklus -------------------------------------------------------

let backendPort = null; // tatsaechlich verwendeter Port (kann vom konfigurierten abweichen)

function backendUrl() {
  return `http://127.0.0.1:${backendPort ?? readBackendPort()}`;
}

/**
 * Health-Check auf einem Port. Ergebnis:
 *  "ours"    – Vid4me-Backend mit UNSEREM Datenordner (wiederverwendbar)
 *  "foreign" – anderes Vid4me-Backend (z.B. Dev-Instanz) oder Fremdprozess
 *  "free"    – keine Antwort
 */
function probePort(port) {
  return new Promise((resolve) => {
    const req = http.get(`http://127.0.0.1:${port}/health`, { timeout: 1000 }, (res) => {
      let body = "";
      res.on("data", (d) => { body += d; });
      res.on("end", () => {
        try {
          const info = JSON.parse(body);
          if (info.app === "vid4me" &&
              path.resolve(info.data_root ?? "") === path.resolve(DATA_ROOT)) {
            resolve("ours");
            return;
          }
        } catch { /* kein Vid4me-Backend */ }
        resolve("foreign");
      });
    });
    req.on("error", () => resolve("free"));
    req.on("timeout", () => { req.destroy(); resolve("foreign"); });
  });
}

async function waitForBackend(timeoutMs = 60000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    if ((await probePort(backendPort ?? readBackendPort())) === "ours") return true;
    await new Promise((r) => setTimeout(r, 700));
  }
  return false;
}

async function startBackend() {
  // Freien bzw. eigenen Port suchen: NIE ein Backend mit fremdem Datenordner
  // mitbenutzen (sonst landen Downloads im falschen Ordner).
  const basePort = readBackendPort();
  for (let i = 0; i < 10; i++) {
    const candidate = basePort + i;
    const state = await probePort(candidate);
    if (state === "ours") {
      backendPort = candidate;
      process.env.VID4ME_PORT = String(candidate);
      console.log("[vid4me] Eigenes Backend laeuft bereits auf Port", candidate);
      return;
    }
    if (state === "free") {
      backendPort = candidate;
      process.env.VID4ME_PORT = String(candidate);
      break;
    }
    console.log(`[vid4me] Port ${candidate} gehoert einer anderen Instanz — weiche aus.`);
  }
  if (backendPort === null) {
    backendPort = basePort;
    process.env.VID4ME_PORT = String(basePort);
  }

  const py = pythonExecutable();
  console.log(`[vid4me] Starte Backend auf Port ${backendPort}:`, py, "-m backend.main");
  backendProc = spawn(py, ["-m", "backend.main"], {
    cwd: APP_ROOT,
    env: { ...process.env, VID4ME_ROOT: DATA_ROOT, VID4ME_PORT: String(backendPort) },
    stdio: ["ignore", "pipe", "pipe"],
    windowsHide: true,
  });
  backendProc.stdout.on("data", (d) => process.stdout.write(`[backend] ${d}`));
  backendProc.stderr.on("data", (d) => process.stderr.write(`[backend] ${d}`));
  backendProc.on("exit", (code) => {
    console.log(`[vid4me] Backend beendet (Code ${code})`);
    backendProc = null;
  });
}

function stopBackend() {
  if (!backendProc) return;
  try {
    // Windows: Prozessbaum beenden (uvicorn spawnt Worker)
    execSync(`taskkill /pid ${backendProc.pid} /T /F`, { windowsHide: true });
  } catch {
    try { backendProc.kill(); } catch { /* bereits beendet */ }
  }
  backendProc = null;
}

// ---- Fenster ---------------------------------------------------------------------

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1480,
    height: 960,
    minWidth: 1100,
    minHeight: 700,
    backgroundColor: "#0d0f14",
    title: "Vid4me",
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
      // Preload braucht fs/path (Port-Config lesen) — die seit Electron 20
      // standardmaessige Renderer-Sandbox stellt kein Node bereit und liess
      // das Preload still scheitern (UI fiel auf Port 8756 zurueck).
      sandbox: false,
    },
  });
}

function loadApp() {
  const devUrl = process.env.VITE_DEV_SERVER_URL;
  if (devUrl) {
    mainWindow.loadURL(devUrl);
  } else {
    mainWindow.loadFile(path.join(__dirname, "..", "dist", "index.html"));
  }

  // Rechtsklick-Menue (Einfuegen etc.) fuer Text-Felder
  mainWindow.webContents.on("context-menu", (_event, params) => {
    const template = [];
    if (params.isEditable) {
      template.push(
        { role: "cut", label: "Ausschneiden", enabled: params.selectionText.length > 0 },
        { role: "copy", label: "Kopieren", enabled: params.selectionText.length > 0 },
        { role: "paste", label: "Einfuegen" },
        { type: "separator" },
        { role: "selectAll", label: "Alles auswaehlen" },
      );
    } else if (params.selectionText.length > 0) {
      template.push({ role: "copy", label: "Kopieren" });
    }
    if (template.length > 0) {
      Menu.buildFromTemplate(template).popup({ window: mainWindow });
    }
  });
}

// ---- App-Lebenszyklus ----------------------------------------------------------------

// Ordner aus der UI im Explorer oeffnen (nur innerhalb des Datenordners)
ipcMain.handle("open-data-folder", (_event, sub) => {
  const target = path.resolve(DATA_ROOT, sub || "");
  if (!target.startsWith(path.resolve(DATA_ROOT))) return false;
  shell.openPath(target);
  return true;
});

app.whenReady().then(async () => {
  if (IS_PACKAGED) ensureDataRoot();
  createWindow();

  if (!fs.existsSync(venvPython()) && IS_PACKAGED) {
    // Ersteinrichtung: Python-Umgebung fehlt noch
    mainWindow.loadFile(path.join(__dirname, "setup.html"));
    const ok = await runFirstTimeSetup();
    if (!ok || !fs.existsSync(venvPython())) {
      dialog.showErrorBox(
        "Vid4me – Einrichtung fehlgeschlagen",
        "Die Python-Umgebung konnte nicht eingerichtet werden.\n\n" +
        `Bitte manuell ausfuehren:\npowershell -ExecutionPolicy Bypass -File "${path.join(APP_ROOT, "scripts", "setup_backend.ps1")}" -DataRoot "${DATA_ROOT}"`
      );
    }
  }

  await startBackend();
  loadApp();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
      loadApp();
    }
  });
});

app.on("window-all-closed", () => {
  stopBackend();
  app.quit();
});

app.on("before-quit", stopBackend);
process.on("exit", stopBackend);
