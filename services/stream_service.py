# services/stream_service.py

import cv2
import time
from typing import Generator
from services.vision_pipeline_service import VisionPipelineService
from utils.logger import get_logger

logger = get_logger(__name__)

class StreamService:
    """
    Modul untuk menghasilkan video stream menggunakan standar MJPEG (Motion JPEG).
    
    Kelas ini dirancang murni untuk mengambil frame visual dari pipeline,
    melakukan encoding (kompresi) menjadi format gambar .jpg, lalu
    menghasilkannya (yield) dalam format byte stream berturut-turut.
    """

    def __init__(self, pipeline_service: VisionPipelineService):
        """
        Inisialisasi StreamService.

        Args:
            pipeline_service (VisionPipelineService): Modul pipeline yang
            menyediakan frame visual (yang sudah diberi anotasi).
        """
        self.pipeline = pipeline_service

    def generate_frames(self) -> Generator[bytes, None, None]:
        """
        Fungsi generator untuk streaming MJPEG.
        
        Fungsi ini berjalan di dalam infinite loop (selama request HTTP hidup).
        Ia akan terus-menerus mengambil frame terbaru, mengonversinya ke JPEG,
        dan merakitnya sesuai struktur multipart/x-mixed-replace.

        Yields:
            bytes: Data bit stream berisi header konten dan gambar JPEG.
        """
        while True:
            # Mengambil tuple hasil dari pipeline (kita abaikan fitur dict-nya)
            _, frame = self.pipeline.get_latest_results()
            
            if frame is None:
                # Jika belum ada frame dari kamera, berikan jeda singkat 
                # agar CPU tidak bekerja 100% untuk perulangan kosong.
                time.sleep(0.01)
                continue
                
            # Mengonversi (encode) matriks gambar BGR OpenCV menjadi format JPEG
            success, buffer = cv2.imencode('.jpg', frame)
            
            if not success:
                logger.error("Gagal melakukan encoding frame ke format JPEG.")
                continue
                
            # Mengubah buffer memori menjadi deretan bytes mentah
            frame_bytes = buffer.tobytes()
            
            # Menghasilkan output yang dibungkus dengan boundary standar MJPEG HTTP
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')