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

        # Menyimpan frame dan distance_json terbaru yang belum diproses
        self._pending_frame: np.ndarray | None = None
        self._pending_distance_json: dict | None = None
        self._pending_robot_id: str | None = None
        self._has_pending = threading.Event()

    def on_robot_frame(self, robot_id: str, frame_bytes: bytes, distance_json: dict) -> None:
        """
        Dipanggil oleh Flask-SocketIO setiap kali robot mengirim frame.
        Menerapkan frame-dropping: hanya simpan frame terbaru, buang yang lama.

        Args:
            robot_id (str): ID unik robot pengirim.
            frame_bytes (bytes): Raw bytes gambar JPEG dari robot.
            distance_json (dict): Payload jarak dari robot: {distance, confidence}.
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

        # Simpan frame terbaru (overwrite frame lama yang belum sempat diproses = frame-drop)
        with self.lock:
            self._pending_frame = frame
            self._pending_distance_json = distance_json
            self._pending_robot_id = robot_id
            self._has_pending.set()

        logger.debug(f"[{robot_id}] Frame diterima. distance={distance_json.get('distance')}")

    def on_frame_array(self, robot_id: str, frame: np.ndarray, distance_json: dict = None) -> None:
        """
        Menyimpan frame numpy array yang sudah terdecode (misal dari raw websocket / camera service).
        """
        if frame is None or frame.size == 0:
            return
        with self.lock:
            self._pending_frame = frame
            self._pending_distance_json = distance_json or {}
            self._pending_robot_id = robot_id
            self._has_pending.set()

    def get_pending(self) -> tuple[str | None, np.ndarray | None, dict | None]:
        """
        Mengambil frame + data terbaru yang menunggu untuk diproses.
        Dipanggil oleh VisionPipelineService dari thread pemprosesannya.

        Returns:
            Tuple (robot_id, frame, distance_json) atau (None, None, None) jika kosong.
        """
        with self.lock:
            if self._pending_frame is None:
                return None, None, None

            robot_id = self._pending_robot_id
            frame = self._pending_frame.copy()
            distance_json = self._pending_distance_json.copy() if self._pending_distance_json else {}

            # Reset pending setelah diambil
            self._pending_frame = None
            self._pending_distance_json = None
            self._pending_robot_id = None
            self._has_pending.clear()

            return robot_id, frame, distance_json

    def wait_for_frame(self, timeout: float = 1.0) -> bool:
        """
        Blocking wait sampai ada frame baru tersedia atau timeout.

        Args:
            timeout (float): Maksimal waktu tunggu dalam detik.

        Returns:
            bool: True jika ada frame, False jika timeout.
        """
        return self._has_pending.wait(timeout=timeout)
