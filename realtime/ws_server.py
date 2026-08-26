# realtime/ws_server.py
"""
ws_server.py — WebSocket server untuk distribusi hasil real-time.

Server berperan ganda:
1. Broadcast real-time — begitu ML Engine menghasilkan status baru,
   langsung dorong ke semua klien yang terhubung.
2. Query riwayat — saat klien (website) mengirim request riwayat,
   server mengambil dari database dan mengirim balik.

Format pesan broadcast (JSON):
    {
        "type": "fatigue" | "dry_eye" | "myopia_risk",
        "status": "...",
        "detail": { ... },
        "timestamp": 1234567890.123
    }

Dua jenis klien:
- Website   → menerima broadcast + mengirim request riwayat.
- Hardware  → menerima broadcast saja (payload ringkas).

Format request riwayat dari klien:
    {
        "action": "get_history",
        "type": "fatigue" | "dry_eye" | "myopia_risk" | "summary",
        "days": 7
    }

Library: websockets (Python, asyncio-native).
"""

import asyncio
import json
import time
from typing import Any, Dict, Optional, Set

import websockets
from websockets.server import serve, WebSocketServerProtocol

from storage.database import HealthDatabase
from utils.logger import get_logger

logger = get_logger(__name__)


class RealtimeWSServer:
    """
    WebSocket server asyncio-native untuk broadcast hasil deteksi
    dan melayani query riwayat dari database.
    """

    def __init__(
        self,
        database: HealthDatabase,
        host: str = "0.0.0.0",
        port: int = 8765,
    ):
        self._db = database
        self._host = host
        self._port = port

        # Set semua klien yang terhubung
        self._clients: Set[WebSocketServerProtocol] = set()
        self._lock = asyncio.Lock()

        # Server instance (diisi saat start)
        self._server: Optional[asyncio.AbstractServer] = None

    # ─────────────────────────────────────────
    # Connection lifecycle
    # ─────────────────────────────────────────

    async def _register(self, ws: WebSocketServerProtocol) -> None:
        async with self._lock:
            self._clients.add(ws)
        client_info = f"{ws.remote_address}" if ws.remote_address else "unknown"
        logger.info(f"[WS] Klien terhubung: {client_info} (total: {len(self._clients)})")

    async def _unregister(self, ws: WebSocketServerProtocol) -> None:
        async with self._lock:
            self._clients.discard(ws)
        client_info = f"{ws.remote_address}" if ws.remote_address else "unknown"
        logger.info(f"[WS] Klien terputus: {client_info} (total: {len(self._clients)})")

    async def _handler(self, ws: WebSocketServerProtocol) -> None:
        """Handler utama per koneksi klien."""
        await self._register(ws)
        try:
            async for message in ws:
                await self._handle_message(ws, message)
        except websockets.ConnectionClosed:
            pass
        except Exception as e:
            logger.error(f"[WS] Error pada handler klien: {e}")
        finally:
            await self._unregister(ws)

    # ─────────────────────────────────────────
    # Message handling (query riwayat)
    # ─────────────────────────────────────────

    async def _handle_message(
        self, ws: WebSocketServerProtocol, raw_message: str
    ) -> None:
        """Proses pesan dari klien (terutama request riwayat)."""
        try:
            msg = json.loads(raw_message)
        except json.JSONDecodeError:
            await ws.send(json.dumps({
                "error": "Format pesan tidak valid (bukan JSON)."
            }))
            return

        action = msg.get("action")

        if action == "get_history":
            await self._handle_history_request(ws, msg)
        elif action == "get_summary":
            await self._handle_summary_request(ws, msg)
        elif action == "ping":
            await ws.send(json.dumps({"action": "pong", "timestamp": time.time()}))
        else:
            await ws.send(json.dumps({
                "error": f"Action tidak dikenal: {action}",
                "supported": ["get_history", "get_summary", "ping"],
            }))

    async def _handle_history_request(
        self, ws: WebSocketServerProtocol, msg: Dict[str, Any]
    ) -> None:
        """Ambil riwayat dari database dan kirim ke klien."""
        history_type = msg.get("type", "fatigue")
        days = int(msg.get("days", 7))

        # Jalankan query DB di thread terpisah (agar tidak blokir event loop)
        loop = asyncio.get_event_loop()

        if history_type == "fatigue":
            data = await loop.run_in_executor(
                None, self._db.get_fatigue_history, days
            )
        elif history_type == "dry_eye":
            data = await loop.run_in_executor(
                None, self._db.get_dry_eye_history, days
            )
        elif history_type == "myopia_risk":
            data = await loop.run_in_executor(
                None, self._db.get_myopia_risk_history, days
            )
        else:
            await ws.send(json.dumps({
                "error": f"Tipe riwayat tidak dikenal: {history_type}",
                "supported": ["fatigue", "dry_eye", "myopia_risk"],
            }))
            return

        response = {
            "action": "history_response",
            "type": history_type,
            "days": days,
            "count": len(data),
            "data": data,
            "timestamp": time.time(),
        }
        await ws.send(json.dumps(response))

    async def _handle_summary_request(
        self, ws: WebSocketServerProtocol, msg: Dict[str, Any]
    ) -> None:
        """Ambil ringkasan statistik dari database."""
        days = int(msg.get("days", 1))

        loop = asyncio.get_event_loop()
        summary = await loop.run_in_executor(
            None, self._db.get_summary, days
        )

        response = {
            "action": "summary_response",
            "data": summary,
            "timestamp": time.time(),
        }
        await ws.send(json.dumps(response))

    # ─────────────────────────────────────────
    # Broadcast (dipanggil dari pipeline thread)
    # ─────────────────────────────────────────

    async def _async_broadcast(self, message: str) -> None:
        """Kirim pesan ke semua klien yang terhubung (async)."""
        async with self._lock:
            clients = self._clients.copy()

        if not clients:
            return

        # Broadcast ke semua klien, abaikan yang gagal
        results = await asyncio.gather(
            *[self._safe_send(ws, message) for ws in clients],
            return_exceptions=True,
        )

        # Bersihkan klien yang disconnect
        for ws, result in zip(clients, results):
            if isinstance(result, Exception):
                await self._unregister(ws)

    async def _safe_send(self, ws: WebSocketServerProtocol, message: str) -> None:
        """Kirim pesan ke satu klien, raise exception jika gagal."""
        try:
            await ws.send(message)
        except websockets.ConnectionClosed:
            raise
        except Exception as e:
            logger.debug(f"[WS] Gagal kirim ke klien: {e}")
            raise

    def broadcast(self, data: Dict[str, Any], loop: asyncio.AbstractEventLoop) -> None:
        """
        Broadcast data ke semua klien (dipanggil dari thread non-async).

        Metode ini thread-safe — menggunakan asyncio.run_coroutine_threadsafe()
        untuk menjadwalkan broadcast di event loop asyncio.

        Args:
            data: Dict yang akan di-serialize ke JSON.
            loop: Event loop asyncio yang sedang berjalan.
        """
        try:
            message = json.dumps(data, default=str)
            asyncio.run_coroutine_threadsafe(
                self._async_broadcast(message), loop
            )
        except Exception as e:
            logger.debug(f"[WS] Error broadcast: {e}")

    def broadcast_results(
        self,
        results: Dict[str, Dict[str, Any]],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """
        Broadcast hasil semua detector sebagai pesan terpisah per tipe.

        Pesan untuk hardware dibuat ringkas (hanya field esensial).

        Args:
            results: Output dari InferenceEngine.run().
            loop: Event loop asyncio.
        """
        for detection_type, result in results.items():
            self.broadcast(result, loop)

    # ─────────────────────────────────────────
    # Server lifecycle
    # ─────────────────────────────────────────

    async def start_async(self) -> None:
        """Mulai WebSocket server (dipanggil dalam asyncio event loop)."""
        self._server = await serve(
            self._handler,
            self._host,
            self._port,
        )
        logger.info(
            f"[WS] WebSocket server berjalan di ws://{self._host}:{self._port}"
        )
        await self._server.wait_closed()

    async def stop_async(self) -> None:
        """Hentikan server secara graceful."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            logger.info("[WS] WebSocket server dihentikan.")

    @property
    def client_count(self) -> int:
        return len(self._clients)
