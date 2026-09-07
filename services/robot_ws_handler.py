# services/robot_ws_handler.py
#
# Modul ini bertanggung jawab menjadi WebSocket SERVER bagi Robot (ESP32-CAM).
# Robot connect ke ML server ini, lalu mengirim frame gambar + distance_json
# setiap kali ada frame baru (15-20fps).
#
# Tugas utama:
#   1. Terima koneksi WebSocket dari Robot
#   2. Decode frame JPEG dari robot menjadi numpy array
#   3. Teruskan frame + distance_json ke VisionPipelineService untuk dianalisa
#   4. Terapkan frame-dropping jika pipeline masih sibuk

import base64
import json
import threading
import numpy as np
import cv2

from utils.logger import get_logger

logger = get_logger(__name__)


class RobotWebSocketHandler:
    """
    Handler untuk koneksi WebSocket dari Robot ESP32-CAM.

    Menerima frame gambar dan distance_json dari robot, lalu
    meneruskannya ke VisionPipelineService untuk dianalisa.
    Frame baru akan di-drop jika pipeline masih memproses frame sebelumnya.
    """

    def __init__(self, pipeline_service):
        """
        Args:
            pipeline_service: Instance VisionPipelineService yang akan
                              memproses tiap frame yang diterima dari robot.
        """
        self.pipeline = pipeline_service
        self.lock = threading.Lock()

        # Multi-robot: satu slot pending per robot_id (round-robin saat diambil).
        # Bentuk: {robot_id: {"frame": np.ndarray, "distance_json": dict, "size": int}}
        self._pending: dict = {}
        self._order: list = []
        self._rr_index: int = 0
        self._has_pending = threading.Event()

    def on_robot_frame(self, robot_id: str, frame_bytes: bytes, distance_json: dict, frame_size_bytes: int = None) -> None:
        """
        Dipanggil oleh Flask-SocketIO setiap kali robot mengirim frame.
        Menerapkan frame-dropping: hanya simpan frame terbaru, buang yang lama.

        Args:
            robot_id (str): ID unik robot pengirim.
            frame_bytes (bytes): Raw bytes gambar JPEG dari robot.
            distance_json (dict): Payload jarak dari robot: {distance, confidence}.
            frame_size_bytes (int, optional): Ukuran data frame dalam bytes.
        """
        # Support both raw bytes & base64 string
        if isinstance(frame_bytes, str):
            try:
                frame_bytes = base64.b64decode(frame_bytes)
            except Exception as e:
                logger.warning(f"[{robot_id}] Gagal decode base64 frame string: {e}")
                return
        elif isinstance(frame_bytes, (bytes, bytearray, memoryview)):
            raw_b = bytes(frame_bytes)
            if raw_b.startswith(b'/9j/'):  # Base64 string encoded as bytes
                try:
                    frame_bytes = base64.b64decode(raw_b)
                except Exception:
                    pass

        # Decode JPEG bytes → numpy array BGR (format OpenCV)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            logger.warning(f"[{robot_id}] Gagal decode frame JPEG dari robot. Frame di-skip.")
            return

        size_bytes = frame_size_bytes if frame_size_bytes is not None else len(frame_bytes)

        # Simpan frame terbaru per-robot (overwrite frame lama robot yang sama = frame-drop).
        # Robot lain tidak tertimpa — masing-masing punya slot sendiri.
        with self.lock:
            self._pending[robot_id] = {
                "frame": frame,
                "distance_json": distance_json,
                "size": size_bytes,
            }
            if robot_id not in self._order:
                self._order.append(robot_id)
            self._has_pending.set()

        logger.debug(f"[{robot_id}] Frame diterima: {size_bytes} bytes ({size_bytes / (1024*1024):.4f} MB). distance={distance_json.get('distance')}")

    def on_frame_array(self, robot_id: str, frame: np.ndarray, distance_json: dict = None, frame_size_bytes: int = None) -> None:
        """
        Menyimpan frame numpy array yang sudah terdecode (misal dari raw websocket / camera service).
        """
        if frame is None or frame.size == 0:
            return
        size_bytes = frame_size_bytes if frame_size_bytes is not None else frame.nbytes
        with self.lock:
            self._pending[robot_id] = {
                "frame": frame,
                "distance_json": distance_json or {},
                "size": size_bytes,
            }
            if robot_id not in self._order:
                self._order.append(robot_id)
            self._has_pending.set()

    def get_pending(self) -> tuple[str | None, np.ndarray | None, dict | None, int]:
        """
        Mengambil satu frame + data terbaru yang menunggu untuk diproses.
        Multi-robot: round-robin antar robot_id agar tidak ada robot yang starvation.
        Dipanggil oleh VisionPipelineService dari thread pemprosesannya.

        Returns:
            Tuple (robot_id, frame, distance_json, frame_size_bytes) atau (None, None, None, 0) jika kosong.
        """
        with self.lock:
            if not self._pending:
                return None, None, None, 0

            # Bersihkan order dari robot yang sudah tidak punya pending
            self._order = [r for r in self._order if r in self._pending]
            for r in self._pending:
                if r not in self._order:
                    self._order.append(r)
            if not self._order:
                return None, None, None, 0

            self._rr_index %= len(self._order)
            robot_id = self._order.pop(self._rr_index)
            # _rr_index tetap (pop menggeser), clamp ke panjang baru
            if self._order:
                self._rr_index %= len(self._order)
            else:
                self._rr_index = 0

            slot = self._pending.pop(robot_id, None)
            if slot is None:
                if not self._pending:
                    self._has_pending.clear()
                return None, None, None, 0

            frame = slot["frame"].copy()
            distance_json = dict(slot["distance_json"]) if slot["distance_json"] else {}
            frame_size_bytes = slot["size"]

            if not self._pending:
                self._has_pending.clear()

            return robot_id, frame, distance_json, frame_size_bytes

    def pending_count(self) -> int:
        """Jumlah robot yang punya frame menunggu diproses."""
        with self.lock:
            return len(self._pending)

    def pending_robots(self) -> list:
        """Daftar robot_id yang punya frame menunggu diproses."""
        with self.lock:
            return list(self._pending.keys())

    def wait_for_frame(self, timeout: float = 1.0) -> bool:
        """
        Blocking wait sampai ada frame baru tersedia atau timeout.

        Args:
            timeout (float): Maksimal waktu tunggu dalam detik.

        Returns:
            bool: True jika ada frame, False jika timeout.
        """
        return self._has_pending.wait(timeout=timeout)
