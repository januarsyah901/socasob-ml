# services/feature_store.py

import threading
from typing import Dict, Any

class FeatureStore:
    """
    Cache in-memory yang thread-safe untuk menyimpan data fitur ekstraksi terbaru.
    
    Kelas ini menyelesaikan masalah konkurensi: VisionPipelineService (background thread) 
    dapat terus-menerus memperbarui data di sini tanpa henti, sementara rute Flask
    (HTTP thread) dapat membaca data kapan saja secara instan tanpa perlu menunggu
    proses Computer Vision selesai.
    """

    def __init__(self):
        """Inisialisasi penyimpan fitur dengan state bawaan dan pengunci (lock)."""
        self.lock = threading.Lock()
        
        # State bawaan jika API dipanggil sebelum frame pertama selesai diproses
        self._features: Dict[str, Any] = {
            "ear": 0.0,
            "eye_status": "Unknown",
            "blink_count": 0,
            "blink_rate": 0.0,
            "eye_closure_duration": 0.0,
            "fps": 0.0,
            "timestamp": "",
            "face_detected": False
        }

    def update(self, new_features: Dict[str, Any]) -> None:
        """
        Memperbarui data fitur dengan hasil ekstraksi terbaru secara aman.

        Args:
            new_features (Dict[str, Any]): Dictionary fitur terbaru dari pipeline.
        """
        with self.lock:
            # Menyimpan salinan agar tidak terjadi mutasi referensi dari luar
            self._features = new_features.copy()

    def get(self) -> Dict[str, Any]:
        """
        Mengambil salinan data fitur terbaru untuk dikirim sebagai JSON API.

        Returns:
            Dict[str, Any]: Salinan dictionary fitur sesuai kontrak API.
        """
        with self.lock:
            return self._features.copy()