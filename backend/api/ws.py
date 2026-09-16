"""WebSocket-Verwaltung: Broadcast von Job-Events an alle verbundenen Clients.

JobManager.broadcast wird aus dem Worker-Thread aufgerufen — Uebergabe an die
asyncio-Loop erfolgt threadsicher ueber run_coroutine_threadsafe.
"""
from __future__ import annotations

import asyncio
import json
from typing import Optional

from fastapi import WebSocket, WebSocketDisconnect


class WsHub:
    def __init__(self):
        self.clients: set[WebSocket] = set()
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop

    async def handle(self, ws: WebSocket) -> None:
        await ws.accept()
        self.clients.add(ws)
        try:
            while True:
                # Client sendet nichts Relevantes; Verbindung offen halten.
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            self.clients.discard(ws)

    async def _send_all(self, text: str) -> None:
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_text(text)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)

    def broadcast_threadsafe(self, event: dict) -> None:
        """Aus beliebigen Threads aufrufbar."""
        if self.loop is None or self.loop.is_closed():
            return
        text = json.dumps(event, ensure_ascii=False, default=str)
        asyncio.run_coroutine_threadsafe(self._send_all(text), self.loop)


hub = WsHub()
