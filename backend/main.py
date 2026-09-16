"""Vid4me Backend — FastAPI-Einstiegspunkt.

Start (durch Electron automatisch, manuell fuer Entwicklung):
    python -m backend.main            # nutzt Port aus config/settings.json
"""
from __future__ import annotations

import asyncio
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Projekt-Root in sys.path, damit "backend.*"-Importe auch bei direktem
# Skriptstart funktionieren.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router
from backend.api.ws import hub
from backend.core.downloads import downloads
from backend.core.jobs import jobs
from backend.core.registry import registry
from backend.core import config


@asynccontextmanager
async def lifespan(app: FastAPI):
    hub.attach_loop(asyncio.get_running_loop())
    jobs.broadcast = hub.broadcast_threadsafe
    downloads.broadcast = hub.broadcast_threadsafe
    registry.scan_loras()
    yield


app = FastAPI(title="Vid4me Backend", lifespan=lifespan)

# Vite-Dev-Server + Electron (file://) zulassen — rein lokale App.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health")
def health():
    # data_root: damit Clients pruefen koennen, ob dieses Backend zu IHREM
    # Datenordner gehoert (verhindert Fehl-Wiederverwendung durch andere Instanzen)
    return {"status": "ok", "app": "vid4me", "data_root": str(config.ROOT)}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await hub.handle(ws)


def main() -> None:
    import os
    import uvicorn
    settings = config.load_settings()
    backend_cfg = settings.get("backend", {})
    # VID4ME_PORT (von Electron gesetzt) hat Vorrang: erlaubt Ausweich-Ports,
    # wenn der konfigurierte Port von einer anderen Instanz belegt ist.
    port = int(os.environ.get("VID4ME_PORT") or backend_cfg.get("port", 8756))
    uvicorn.run(
        app,
        host=backend_cfg.get("host", "127.0.0.1"),
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
