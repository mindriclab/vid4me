const { contextBridge, ipcRenderer } = require("electron");
const fs = require("fs");
const path = require("path");

// Datenordner: vom Main-Process gesetzt (gepackt: %USERPROFILE%\Vid4me,
// dev: Projektordner); Fallback fuer den Dev-Modus.
const DATA_ROOT = process.env.VID4ME_DATA_ROOT
  || path.resolve(__dirname, "..", "..");

function backendPort() {
  // Der Main-Process setzt VID4ME_PORT auf den tatsaechlich verwendeten Port
  // (kann vom konfigurierten abweichen, wenn ausgewichen wurde).
  if (process.env.VID4ME_PORT) return Number(process.env.VID4ME_PORT);
  try {
    const settings = JSON.parse(
      fs.readFileSync(path.join(DATA_ROOT, "config", "settings.json"), "utf-8")
    );
    return settings.backend?.port ?? 8756;
  } catch {
    return 8756;
  }
}

contextBridge.exposeInMainWorld("vid4me", {
  backendUrl: `http://127.0.0.1:${backendPort()}`,
  platform: process.platform,
  // Erst-Setup: Log-Zeilen aus dem Main-Process (setup.html hoert hierauf)
  onSetupLog: (callback) =>
    ipcRenderer.on("setup-log", (_event, line) => callback(line)),
  // Ordner im Explorer oeffnen (relativ zum Datenordner, z.B. "outputs")
  openDataFolder: (sub) => ipcRenderer.invoke("open-data-folder", sub),
});
