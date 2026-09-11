# services/robot_trigger_service.py
"""
robot_trigger_service.py — Pengelola Trigger Pesan Teks ke Robot ESP32.

Mengelola koneksi aktif raw WebSocket dari ESP32-CAM dan mengirimkan
trigger ekspresi berupa teks polos:
- "normal" : Kondisi mata sehat / santai
- "5"      : Mata mulai lelah (tahap 1, 5 menit)
- "10"     : Mata lelah berat / kronis (>= 10 menit)
- "dry"    : Terdeteksi mata kering
- "20"     : Peringatan istirahat aturan 20-20-20
"""

import threading
import time
from typing import Any, Dict, Optional
from utils.logger import get_logger

logger = get_logger(__name__)

VALID_TRIGGERS = {"normal", "5", "10", "dry", "20"}


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
        self._manual_override_until: Dict[str, float] = {}
        self._manual_override_payloads: Dict[str, dict] = {}
        self._global_lock = threading.Lock()

    def set_be_client(self, be_socket_client) -> None:
        """Set atau update instance be_socket_client."""
        self.be_client = be_socket_client

    def register_connection(self, robot_id: str, ws: Any) -> None:
        """
        Mendaftarkan koneksi WebSocket aktif untuk sebuah robot_id.
        Mengirimkan trigger baseline awal jika belum ada trigger aktif atau override.
        """
        if not robot_id or ws is None:
            return

        with self._global_lock:
            if self._connections.get(robot_id) == ws:
                return

            self._connections[robot_id] = ws
            if robot_id not in self._locks:
                self._locks[robot_id] = threading.Lock()

        logger.info(f"[TriggerService] Robot '{robot_id}' terdaftar di WebSocket trigger manager.")

        # Jangan paksa 'normal' jika robot sedang dalam masa manual override
        if not self.is_in_manual_override(robot_id):
            self.send_trigger(robot_id, "normal", force=True)
        else:
            override_meta = self.get_override_payload(robot_id) or {}
            trig = override_meta.get("trigger", self.get_last_trigger(robot_id))
            self.send_trigger(robot_id, trig, force=True)

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

    def get_connected_robots(self) -> list:
        """Daftar robot_id yang sedang terhubung via WebSocket."""
        with self._global_lock:
            return list(self._connections.keys())

    def is_in_manual_override(self, robot_id: Optional[str] = None) -> bool:
        """Cek apakah robot sedang dalam masa manual override (holding)."""
        now = time.time()
        with self._global_lock:
            if self._manual_override_until.get("__all__", 0.0) > now:
                return True
            if robot_id and self._manual_override_until.get(robot_id, 0.0) > now:
                return True
            return False

    def get_manual_override_remaining(self, robot_id: Optional[str] = None) -> float:
        """Sisa durasi manual override dalam detik."""
        now = time.time()
        rem = 0.0
        with self._global_lock:
            all_rem = self._manual_override_until.get("__all__", 0.0) - now
            if all_rem > rem:
                rem = all_rem
            if robot_id:
                rid_rem = self._manual_override_until.get(robot_id, 0.0) - now
                if rid_rem > rem:
                    rem = rid_rem
        return max(0.0, rem)

    def get_override_payload(self, robot_id: Optional[str] = None) -> Optional[dict]:
        """Ambil metadata payload override aktif (jika ada)."""
        if not self.is_in_manual_override(robot_id):
            return None
        with self._global_lock:
            if robot_id and robot_id in self._manual_override_payloads:
                return self._manual_override_payloads[robot_id].copy()
            if "__all__" in self._manual_override_payloads:
                return self._manual_override_payloads["__all__"].copy()
            return None

    def set_manual_override(
        self,
        robot_id: str,
        trigger: str,
        duration_sec: float = 20.0,
        meta: Optional[dict] = None
    ) -> bool:
        """
        Kirim trigger paksa dan tahan status tersebut selama duration_sec detik.
        Selama durasi ini, trigger otomatis dari pipeline vision akan diabaikan.
        """
        trigger_str = str(trigger).strip().lower()
        if trigger_str not in VALID_TRIGGERS:
            trigger_str = "normal"

        now = time.time()
        until = now + max(0.0, duration_sec)
        with self._global_lock:
            self._manual_override_until[robot_id] = until
            payload = meta.copy() if meta else {}
            payload["trigger"] = trigger_str
            self._manual_override_payloads[robot_id] = payload

        logger.info(f"[TriggerService] Manual override robot '{robot_id}' -> '{trigger_str}' selama {duration_sec}s.")
        return self.send_trigger(robot_id, trigger_str, force=True)

    def broadcast_manual_override(
        self,
        trigger: str,
        duration_sec: float = 20.0,
        meta: Optional[dict] = None
    ) -> int:
        """
        Broadcast trigger paksa dan tahan status ke semua robot selama duration_sec detik.
        """
        trigger_str = str(trigger).strip().lower()
        if trigger_str not in VALID_TRIGGERS:
            trigger_str = "normal"

        now = time.time()
        until = now + max(0.0, duration_sec)
        with self._global_lock:
            self._manual_override_until["__all__"] = until
            payload = meta.copy() if meta else {}
            payload["trigger"] = trigger_str
            self._manual_override_payloads["__all__"] = payload

            robot_ids = list(self._connections.keys())
            for rid in robot_ids:
                self._manual_override_until[rid] = until
                self._manual_override_payloads[rid] = payload.copy()

        logger.info(f"[TriggerService] Manual override broadcast -> '{trigger_str}' selama {duration_sec}s.")
        return self.broadcast_trigger(trigger_str, force=True)

    def clear_manual_override(self, robot_id: Optional[str] = None) -> None:
        """Hapus status manual override agar sistem kembali otomatis ke pipeline AI."""
        with self._global_lock:
            if robot_id:
                self._manual_override_until.pop(robot_id, None)
                self._manual_override_payloads.pop(robot_id, None)
            else:
                self._manual_override_until.clear()
                self._manual_override_payloads.clear()
        logger.info(f"[TriggerService] Manual override dibersihkan untuk '{robot_id or 'all'}'.")

    def get_last_trigger(self, robot_id: str) -> str:
        """Mengambil trigger terakhir yang dikirim ke robot (default: 'normal')."""
        return self._last_triggers.get(robot_id, "normal")

    def send_trigger(self, robot_id: str, trigger: str, force: bool = False) -> bool:
        """
        Kirim trigger pesan teks polos ("normal", "5", "10", "dry", "20") ke robot.

        Args:
            robot_id: ID unik robot target.
            trigger: Nilai trigger ("normal", "5", "10", "dry", "20").
            force: Jika True, paksa kirim meski sama dengan trigger sebelumnya atau dalam masa override.

        Returns:
            bool: True jika berhasil terkirim ke socket robot, False jika dilewati / gagal.
        """
        # Normalisasi trigger
        trigger_str = str(trigger).strip().lower()
        if trigger_str not in VALID_TRIGGERS:
            logger.warning(f"[TriggerService] Trigger '{trigger}' tidak dikenal, fallback ke 'normal'.")
            trigger_str = "normal"

        now = time.time()

        # Abaikan trigger otomatis jika robot sedang dalam masa manual override (holding)
        if not force and self.is_in_manual_override(robot_id):
            return False

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
            "20": "break_20m",
        }
        return mapping.get(trigger, "normal")
