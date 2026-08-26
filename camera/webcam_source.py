# camera/webcam_source.py
"""
webcam_source.py — Implementasi CameraSource untuk webcam lokal via OpenCV.

Menangani koneksi, pembacaan frame, dan pembersihan resource
khusus untuk perangkat kamera USB/Internal.
"""

from typing import Tuple

import cv2
import numpy as np

from camera.base import CameraSource
from utils.logger import get_logger

logger = get_logger(__name__)


class WebcamSource(CameraSource):
    """
    Sumber video dari webcam lokal menggunakan OpenCV VideoCapture.
    """

    def __init__(self, camera_index: int = 0):
        """
        Args:
            camera_index: Index device kamera pada OS (default: 0).
        """
        logger.info(f"Menginisialisasi WebcamSource pada index {camera_index}")
        self._cap = cv2.VideoCapture(camera_index)

        if not self._cap.isOpened():
            logger.error(f"Gagal membuka webcam dengan index {camera_index}")

    def read_frame(self) -> np.ndarray | None:
        if not self.is_opened():
            return None

        ret, frame = self._cap.read()
        if not ret:
            logger.warning("Gagal menangkap frame dari webcam.")
            return None

        return frame

    def get_fps(self) -> float:
        if not self.is_opened():
            return 0.0
        return self._cap.get(cv2.CAP_PROP_FPS)

    def get_resolution(self) -> Tuple[int, int]:
        """Mengembalikan (width, height) dari properti hardware webcam."""
        if not self.is_opened():
            return (0, 0)
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)

    def is_opened(self) -> bool:
        return self._cap.isOpened()

    def release(self) -> None:
        logger.info("Melepaskan resource WebcamSource")
        if self._cap.isOpened():
            self._cap.release()
