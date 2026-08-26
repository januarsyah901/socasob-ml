# camera/esp32_source.py
"""
esp32_source.py — Implementasi CameraSource untuk ESP32-CAM (HTTP MJPEG stream).

Mendukung dua mode:
1. HTTP MJPEG stream: OpenCV VideoCapture langsung ke URL stream.
2. WebSocket push: frame didorong dari luar via method push_frame()
   (misal dari server WebSocket yang menerima raw binary JPEG).
"""

import threading
import time
from typing import Optional, Tuple
from urllib.parse import urlparse

import cv2
import numpy as np

from camera.base import CameraSource
from utils.logger import get_logger

logger = get_logger(__name__)


def decode_websocket_packet(
    packet: bytes,
    device_id_size: int = 16,
) -> Optional[Tuple[str, np.ndarray]]:
    """
    Decode paket biner dari ESP32-CAM.

    Mendukung dua format:
    - Raw JPEG (dimulai dengan magic bytes 0xFFD8)
    - Legacy format: [device_id (N bytes)] + [JPEG data]

    Returns:
        Tuple (device_id, frame_bgr) atau None jika gagal decode.
    """
    if packet.startswith(b"\xff\xd8"):
        frame = cv2.imdecode(
            np.frombuffer(packet, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        return ("UNKNOWN", frame) if frame is not None else None

    if len(packet) <= device_id_size:
        return None

    device_id = packet[:device_id_size].rstrip(b"\x00").decode(
        "ascii", errors="replace"
    )
    frame = cv2.imdecode(
        np.frombuffer(packet[device_id_size:], dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if frame is None:
        return None
    return device_id, frame


class ESP32CamSource(CameraSource):
    """
    Sumber video dari ESP32-CAM via HTTP MJPEG stream atau WebSocket push.
    """

    def __init__(self, stream_url: str):
        """
        Args:
            stream_url: URL stream (http:// untuk MJPEG, ws:// untuk WebSocket mode).
        """
        self._stream_url = stream_url
        self._capture: Optional[cv2.VideoCapture] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._last_frame_time: Optional[float] = None
        self._fps = 0.0
        self._resolution: Tuple[int, int] = (640, 480)  # default ESP32-CAM
        self._is_websocket_source = urlparse(stream_url).scheme.lower() in {"ws", "wss"}

        if self._is_websocket_source:
            logger.info("ESP32CamSource menunggu frame WebSocket dari: %s", stream_url)
        else:
            logger.info("Membuka HTTP stream ESP32-CAM: %s", stream_url)
            self._capture = cv2.VideoCapture(stream_url)
            if not self._capture.isOpened():
                logger.error("Gagal membuka HTTP stream ESP32-CAM.")
            else:
                w = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if w > 0 and h > 0:
                    self._resolution = (w, h)

    def read_frame(self) -> Optional[np.ndarray]:
        if self._is_websocket_source:
            with self._lock:
                if self._latest_frame is not None:
                    return self._latest_frame.copy()
                return None

        if self._capture is None or not self._capture.isOpened():
            return None

        success, frame = self._capture.read()
        if not success:
            logger.warning("Gagal membaca frame dari HTTP stream ESP32-CAM.")
            time.sleep(0.01)
            return None

        # Update resolusi dari frame aktual
        h, w = frame.shape[:2]
        self._resolution = (w, h)
        self._update_fps()
        return frame

    def push_frame(self, frame: np.ndarray) -> None:
        """
        Menyimpan frame yang didekode dari WebSocket untuk dikonsumsi pipeline.

        Args:
            frame: Frame BGR numpy array.

        Raises:
            ValueError: Jika frame kosong.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Frame ESP32-CAM tidak boleh kosong.")
        with self._lock:
            self._latest_frame = frame.copy()
            h, w = frame.shape[:2]
            self._resolution = (w, h)
        self._update_fps()

    def get_fps(self) -> float:
        if self._capture is not None:
            capture_fps = self._capture.get(cv2.CAP_PROP_FPS)
            if capture_fps > 0:
                return float(capture_fps)
        with self._lock:
            return self._fps

    def get_resolution(self) -> Tuple[int, int]:
        """Mengembalikan (width, height) dari frame terakhir atau default ESP32-CAM."""
        return self._resolution

    def is_opened(self) -> bool:
        if self._is_websocket_source:
            return True
        return self._capture is not None and self._capture.isOpened()

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
        with self._lock:
            self._latest_frame = None

    def _update_fps(self) -> None:
        now = time.monotonic()
        with self._lock:
            if self._last_frame_time is not None:
                interval = now - self._last_frame_time
                if interval > 0:
                    self._fps = 1.0 / interval
            self._last_frame_time = now
