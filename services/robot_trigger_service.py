# services/robot_trigger_service.py
"""
robot_trigger_service.py — Pengelola Trigger Pesan Teks ke Robot ESP32.

Mengelola koneksi aktif raw WebSocket dari ESP32-CAM dan mengirimkan
trigger ekspresi berupa teks polos:
- "normal" : Kondisi mata sehat / santai
- "5"      : Mata mulai lelah (tahap 1, 5 menit)
- "10"     : Mata lelah berat / kronis (>= 10 menit)
- "dry"    : Terdeteksi mata kering
"""

import threading
import time
from typing import Any, Dict, Optional
from utils.logger import get_logger

logger = get_logger(__name__)

VALID_TRIGGERS = {"normal", "5", "10", "dry"}


class RobotTriggerService:
    """
    Service thread-safe untuk mengelola pengiriman pesan teks trigger ke robot.
    """

    def __init__(self, be_socket_client=None):
        self.be_client = be_socket_client
        self._connections: Dict[str, Any] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._last_triggers: Dict[str, str] = {}
        self._last_sent_times: Dict[str, float] = {}
        self._global_lock = threading.Lock()

    def set_be_client(self, be_socket_client) -> None:
        """Set atau update instance be_socket_client."""
        self.be_client = be_socket_client

    def register_connection(self, robot_id: str, ws: Any) -> None:
        """
        Mendaftarkan koneksi WebSocket aktif untuk sebuah robot_id.
        Mengirimkan trigger baseline awal ("normal").
        """
        if not robot_id or ws is None:
            return

        with self._global_lock:
            self._connections[robot_id] = ws
            if robot_id not in self._locks:
                self._locks[robot_id] = threading.Lock()

        logger.info(f"[TriggerService] Robot '{robot_id}' terdaftar di WebSocket trigger manager.")

        # Kirim baseline awal "normal" saat pertama terhubung
        self.send_trigger(robot_id, "normal", force=True)

    def unregister_connection(self, robot_id: str, ws: Any = None) -> None:
        """
        Menghapus koneksi WebSocket yang terputus.
        """
        with self._global_lock:
            current_ws = self._connections.get(robot_id)
            if ws is None or current_ws == ws:
                self._connections.pop(robot_id, None)
                self._locks.pop(robot_id, None)
                logger.info(f"[TriggerService] Robot '{robot_id}' dihapus dari WebSocket trigger manager.")

    def is_connected(self, robot_id: str) -> bool:
        """Cek apakah robot sedang aktif terhubung via WebSocket."""
        with self._global_lock:
            return robot_id in self._connections

    def get_last_trigger(self, robot_id: str) -> str:
        """Mengambil trigger terakhir yang dikirim ke robot (default: 'normal')."""
        return self._last_triggers.get(robot_id, "normal")

    def send_trigger(self, robot_id: str, trigger: str, force: bool = False) -> bool:
        """
        Kirim trigger pesan teks polos ("normal", "5", "10", "dry") ke robot.

        Args:
            robot_id: ID unik robot target.
            trigger: Nilai trigger ("normal", "5", "10", "dry").
            force: Jika True, paksa kirim meski sama dengan trigger sebelumnya.

        Returns:
            bool: True jika berhasil terkirim ke socket robot, False jika dilewati / gagal.
        """
        # Normalisasi trigger
        trigger_str = str(trigger).strip().lower()
        if trigger_str not in VALID_TRIGGERS:
            logger.warning(f"[TriggerService] Trigger '{trigger}' tidak dikenal, fallback ke 'normal'.")
            trigger_str = "normal"

        now = time.time()

        # State-change debouncing (hanya kirim jika berbeda atau forced)
        last_trigger = self._last_triggers.get(robot_id)
        if not force and last_trigger == trigger_str:
            return False

        # Ambil socket & lock
        ws = None
        ws_lock = None
        with self._global_lock:
            ws = self._connections.get(robot_id)
            ws_lock = self._locks.get(robot_id)

        sent_to_hardware = False
        if ws is not None and ws_lock is not None:
            try:
                with ws_lock:
                    ws.send(trigger_str)
                sent_to_hardware = True
                self._last_triggers[robot_id] = trigger_str
                self._last_sent_times[robot_id] = now
                logger.info(f"[TriggerService] -> Pesan teks '{trigger_str}' berhasil dikirim ke robot '{robot_id}'.")
            except Exception as e:
                logger.warning(f"[TriggerService] Gagal kirim trigger ke robot '{robot_id}': {e}")
                self.unregister_connection(robot_id, ws)
        else:
            # Tetap simpan state trigger terakhir meskipun hardware belum tersambung
            self._last_triggers[robot_id] = trigger_str
            self._last_sent_times[robot_id] = now
            logger.debug(f"[TriggerService] Robot '{robot_id}' tidak memiliki socket aktif. State disimpan: '{trigger_str}'.")

        # Teruskan update ke Backend (Socket.io room robot)
        if self.be_client:
            try:
                self.be_client.emit_hardware_status({
                    "robot_id": robot_id,
                    "robot_trigger": trigger_str,
                    "lcd_command": self._trigger_to_lcd_cmd(trigger_str),
                    "connected_hardware": sent_to_hardware,
                    "timestamp": time.time(),
                })
            except Exception as e:
                logger.debug(f"[TriggerService] Gagal emit status ke BE: {e}")

        return sent_to_hardware

    def broadcast_trigger(self, trigger: str, force: bool = False) -> int:
        """Kirim trigger ke seluruh robot yang sedang aktif terhubung."""
        with self._global_lock:
            robot_ids = list(self._connections.keys())

        count = 0
        for rid in robot_ids:
            if self.send_trigger(rid, trigger, force=force):
                count += 1
        return count

    @staticmethod
    def _trigger_to_lcd_cmd(trigger: str) -> str:
        """Helper pemetaan balik dari trigger text ke command LCD."""
        mapping = {
            "normal": "normal",
            "5": "fatigue_5m",
            "10": "fatigue_10m",
            "dry": "dry_eye",
        }
        return mapping.get(trigger, "normal")
