# camera/camera_factory.py

from camera.base_camera import BaseCamera
from camera.webcam_camera import WebcamCamera
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

class CameraFactory:
    """
    Factory class untuk menginisialisasi sumber kamera.
    
    Menggunakan Factory Pattern agar penentuan jenis kamera (webcam vs ESP32)
    terpusat di satu tempat dan dikendalikan sepenuhnya oleh config. Modul lain
    cukup memanggil CameraFactory.get_camera() tanpa perlu tahu implementasi aslinya.
    """

    @staticmethod
    def get_camera() -> BaseCamera:
        """
        Membuat dan mengembalikan instance kamera berdasarkan konfigurasi VIDEO_SOURCE.

        Returns:
            BaseCamera: Instance dari implementasi BaseCamera (WebcamCamera atau ESP32Camera).
        """
        source = settings.VIDEO_SOURCE.lower()
        
        if source == "webcam":
            logger.info("Factory memilih WebcamCamera sebagai sumber video.")
            return WebcamCamera(camera_index=settings.WEBCAM_INDEX)
            
        elif source == "esp32":
            logger.info("Factory memilih ESP32Camera sebagai sumber video.")
            # TODO: Di-uncomment saat esp32_camera.py sudah diimplementasikan di akhir
            # from camera.esp32_camera import ESP32Camera
            # return ESP32Camera(stream_url=settings.ESP32_STREAM_URL)
            raise NotImplementedError("ESP32Camera belum diimplementasikan. Gunakan 'webcam' di config.")
            
        else:
            logger.error(f"VIDEO_SOURCE tidak dikenal: {source}. Fallback ke WebcamCamera.")
            return WebcamCamera(camera_index=settings.WEBCAM_INDEX)