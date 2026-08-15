# services/camera_service.py

import threading
import time
import numpy as np
from typing import Optional
from camera.camera_factory import CameraFactory
from utils.logger import get_logger

logger = get_logger(__name__)

class CameraService:
    """
    Mengelola lifecycle kamera dan background thread untuk capture frame.
    
    Membaca frame secara konstan di thread terpisah agar tidak memblokir
    request HTTP dari Flask. Menggunakan threading.Lock untuk memastikan
    frame terbaru aman dibaca oleh thread lain.
    """

    def __init__(self):
        """
        Inisialisasi CameraService menggunakan CameraFactory.
        """
        self.camera = CameraFactory.get_camera()
        self.latest_frame: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.is_running = False
        self.thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """
        Memulai background thread untuk membaca frame dari kamera.
        """
        if self.is_running:
            logger.warning("CameraService sudah berjalan.")
            return

        if not self.camera.is_opened():
            logger.error("Kamera tidak dapat dibuka. Thread tidak dijalankan.")
            return

        self.is_running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        logger.info("Background thread CameraService berhasil dimulai.")

    def _capture_loop(self) -> None:
        """
        Loop internal yang berjalan di background thread untuk mengambil frame
        secepat mungkin tanpa memblokir thread utama Flask.
        """
        while self.is_running:
            frame = self.camera.read_frame()
            
            if frame is not None:
                # Gunakan lock saat menulis ke shared memory
                with self.lock:
                    self.latest_frame = frame
            else:
                # Beri sedikit jeda jika kamera gagal membaca agar CPU tidak 100%
                time.sleep(0.01)

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """
        Mengambil salinan frame terbaru secara thread-safe.

        Returns:
            Optional[np.ndarray]: Matriks BGR terakhir, atau None jika belum ada.
        """
        with self.lock:
            if self.latest_frame is not None:
                # Selalu kembalikan salinan (copy) agar thread pembaca 
                # tidak memodifikasi array memori yang sama
                return self.latest_frame.copy()
            return None

    def stop(self) -> None:
        """
        Menghentikan thread dan melepaskan resource hardware kamera.
        """
        logger.info("Menghentikan CameraService...")
        self.is_running = False
        
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            
        self.camera.release()
        logger.info("CameraService berhasil dihentikan.")