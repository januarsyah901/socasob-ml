# camera/webcam_camera.py

import cv2
import numpy as np
from camera.base_camera import BaseCamera
from utils.logger import get_logger

logger = get_logger(__name__)

class WebcamCamera(BaseCamera):
    """
    Implementasi sumber video menggunakan webcam lokal via OpenCV.
    
    Kelas ini menangani koneksi, pembacaan frame, dan pembersihan resource
    khusus untuk perangkat kamera USB/Internal.
    """

    def __init__(self, camera_index: int = 0):
        """
        Inisialisasi koneksi ke webcam lokal.

        Args:
            camera_index (int): Index device kamera pada sistem operasi (default: 0).
        """
        logger.info(f"Menginisialisasi WebcamCamera pada index {camera_index}")
        self.cap = cv2.VideoCapture(camera_index)
        
        if not self.cap.isOpened():
            logger.error(f"Gagal membuka webcam dengan index {camera_index}")

    def read_frame(self) -> np.ndarray | None:
        """
        Membaca frame terbaru dari webcam.

        Returns:
            np.ndarray | None: Matriks gambar BGR berupa array NumPy jika berhasil membaca frame.
                               Mengembalikan None jika gagal membaca.
        """
        if not self.is_opened():
            return None
            
        ret, frame = self.cap.read()
        if not ret:
            logger.warning("Gagal menangkap frame dari webcam.")
            return None
            
        return frame

    def get_fps(self) -> float:
        """
        Mendapatkan properti Frame Per Second (FPS) dari hardware webcam.

        Returns:
            float: Nilai FPS dari hardware, atau 0.0 jika tidak tersedia/terputus.
        """
        if not self.is_opened():
            return 0.0
        return self.cap.get(cv2.CAP_PROP_FPS)

    def is_opened(self) -> bool:
        """
        Mengecek apakah koneksi ke hardware webcam masih aktif.

        Returns:
            bool: True jika webcam siap, False sebaliknya.
        """
        return self.cap.isOpened()

    def release(self) -> None:
        """
        Menutup koneksi ke hardware webcam dan membersihkan resource OpenCV.
        """
        logger.info("Melepaskan resource WebcamCamera")
        if self.cap.isOpened():
            self.cap.release()