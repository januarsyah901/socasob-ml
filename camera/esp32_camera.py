import threading
import time
from typing import Optional, Tuple
from urllib.parse import urlparse

import cv2
import numpy as np

from camera.base_camera import BaseCamera
from utils.logger import get_logger

logger = get_logger(__name__)


def decode_websocket_packet(
    packet: bytes,
    device_id_size: int = 16,
) -> Optional[Tuple[str, np.ndarray]]:
    """Decode a raw JPEG or legacy ID-prefixed ESP32 packet."""
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


class ESP32Camera(BaseCamera):
    """Camera adapter for HTTP streams and pushed raw WebSocket frames."""

    def __init__(self, stream_url: str):
        self.stream_url = stream_url
        self._capture: Optional[cv2.VideoCapture] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._last_frame_time: Optional[float] = None
        self._fps = 0.0
        self._is_websocket_source = urlparse(stream_url).scheme.lower() in {"ws", "wss"}

        if self._is_websocket_source:
            logger.info("ESP32Camera menunggu frame WebSocket dari server: %s", stream_url)
        else:
            logger.info("Membuka HTTP stream ESP32-CAM: %s", stream_url)
            self._capture = cv2.VideoCapture(stream_url)
            if not self._capture.isOpened():
                logger.error("Gagal membuka HTTP stream ESP32-CAM.")

    def read_frame(self) -> Optional[np.ndarray]:
        """Read the latest frame from HTTP/MJPEG or the WebSocket buffer."""
        if self._is_websocket_source:
            with self._lock:
                return self._latest_frame.copy() if self._latest_frame is not None else None

        if self._capture is None or not self._capture.isOpened():
            return None

        success, frame = self._capture.read()
        if not success:
            logger.warning("Gagal membaca frame dari HTTP stream ESP32-CAM.")
            time.sleep(0.01)
            return None
        self._update_fps()
        return frame

    def push_frame(self, frame: np.ndarray) -> None:
        """Store a decoded WebSocket frame for the camera service to consume."""
        if frame is None or frame.size == 0:
            raise ValueError("Frame ESP32-CAM tidak boleh kosong.")
        with self._lock:
            self._latest_frame = frame.copy()
        self._update_fps()

    def get_fps(self) -> float:
        """Return the source FPS when available, otherwise the measured FPS."""
        if self._capture is not None:
            capture_fps = self._capture.get(cv2.CAP_PROP_FPS)
            if capture_fps > 0:
                return float(capture_fps)
        with self._lock:
            return self._fps

    def is_opened(self) -> bool:
        """Return whether the HTTP stream is open or WebSocket mode is ready."""
        if self._is_websocket_source:
            return True
        return self._capture is not None and self._capture.isOpened()

    def release(self) -> None:
        """Release the HTTP stream and clear any buffered WebSocket frame."""
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