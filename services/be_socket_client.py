# services/be_socket_client.py
#
# Modul ini bertanggung jawab menjadi Socket.io CLIENT yang terkoneksi ke Backend (BE).
# ML aktif mengirim (push) data ke BE — bukan menunggu dipoll.
#
# Ada dua jenis event yang dikirim ke BE:
#   - py-eye-detection   → real-time tiap frame (distance + blink_event)
#   - py-minute-summary  → agregasi tiap 1 menit (statistik lengkap)

import threading
import socketio

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class BackendSocketClient:
    """
    Socket.io client yang menghubungkan ML ke Backend (Node.js).

    Menggunakan reconnection otomatis sehingga jika BE restart,
    ML akan terkoneksi kembali tanpa perlu restart manual.
    """

    def __init__(self):
        self._sio = socketio.Client(
            reconnection=True,
            reconnection_attempts=0,       # 0 = coba terus selamanya
            reconnection_delay=2,          # tunggu 2 detik sebelum retry
            reconnection_delay_max=10,     # maksimal tunggu 10 detik
            logger=False,
            engineio_logger=False
        )
        self._connected = False
        self._lock = threading.Lock()

        # Daftarkan event handlers
        self._sio.on('connect', self._on_connect)
        self._sio.on('disconnect', self._on_disconnect)
        self._sio.on('connect_error', self._on_connect_error)

    # ------------------------------------------------------------------
    # Internal Event Handlers
    # ------------------------------------------------------------------

    def _on_connect(self):
        self._connected = True
        logger.info(f"[BE Client] Terhubung ke Backend di {settings.BE_URL}")

    def _on_disconnect(self):
        self._connected = False
        logger.warning("[BE Client] Terputus dari Backend. Mencoba reconnect...")

    def _on_connect_error(self, data):
        self._connected = False
        logger.error(f"[BE Client] Gagal terhubung ke Backend: {data}")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect_async(self) -> None:
        """
        Memulai koneksi ke BE di thread terpisah agar tidak blocking.
        """
        def _connect():
            try:
                logger.info(f"[BE Client] Menghubungkan ke {settings.BE_URL} ...")
                self._sio.connect(
                    settings.BE_URL,
                    transports=['websocket', 'polling'],
                    wait_timeout=10
                )
                self._sio.wait()  # Blokir thread ini agar koneksi tetap hidup
            except Exception as e:
                logger.error(f"[BE Client] Error saat connect: {e}")

        thread = threading.Thread(target=_connect, daemon=True, name="be-socket-client")
        thread.start()

    def disconnect(self) -> None:
        """Memutuskan koneksi ke BE secara bersih."""
        if self._sio.connected:
            self._sio.disconnect()

    # ------------------------------------------------------------------
    # Emit Methods (Channel A & B)
    # ------------------------------------------------------------------

    def emit_realtime(self, robot_id: str, distance: str, confidence: int,
                      blink_event: bool, timestamp: str) -> None:
        """
        CHANNEL A — Kirim data real-time tiap frame ke BE.
        Event: py-eye-detection

        Args:
            robot_id (str): ID unik robot.
            distance (str): "Dekat" atau "Jauh".
            confidence (int): Tingkat keyakinan (0-100).
            blink_event (bool): True jika frame ini mendeteksi satu kedipan penuh.
            timestamp (str): Waktu deteksi dalam format ISO 8601.
        """
        if not self._connected:
            return

        payload = {
            "robot_id": robot_id,
            "distance": distance,
            "confidence": confidence,
            "blink_event": blink_event,
            "timestamp": timestamp
        }

        try:
            self._sio.emit('py-eye-detection', payload)
            logger.debug(f"[BE Client] emit py-eye-detection → {distance} | blink={blink_event}")
        except Exception as e:
            logger.error(f"[BE Client] Gagal emit py-eye-detection: {e}")

    def emit_minute_summary(self, summary: dict) -> None:
        """
        CHANNEL B — Kirim ringkasan 1 menit ke BE.
        Event: py-minute-summary

        Args:
            summary (dict): Payload lengkap hasil agregasi 60 detik dari AggregatorService.
        """
        if not self._connected:
            logger.warning("[BE Client] Tidak terhubung. py-minute-summary dibuang.")
            return

        try:
            self._sio.emit('py-minute-summary', summary)
            logger.info(f"[BE Client] emit py-minute-summary → robot={summary.get('robot_id')}")
        except Exception as e:
            logger.error(f"[BE Client] Gagal emit py-minute-summary: {e}")

    @property
    def is_connected(self) -> bool:
        return self._connected
