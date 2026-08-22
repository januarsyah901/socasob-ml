# services/feature_store.py

import threading
from typing import Dict, Any, Optional

class FeatureStore:
    """
    Cache in-memory yang thread-safe untuk menyimpan data fitur ekstraksi terbaru.
    
    Kelas ini menyelesaikan masalah konkurensi: VisionPipelineService (background thread) 
    dapat terus-menerus memperbarui data di sini tanpa henti, sementara rute Flask
    (HTTP thread) dapat membaca data kapan saja secara instan tanpa perlu menunggu
    proses Computer Vision selesai.
    
    Jika belum ada robot yang connect dan mengirim frame, `get()` mengembalikan None
    sehingga API tidak menampilkan data palsu/dummy.
    """

    def __init__(self):
        """Inisialisasi penyimpan fitur dalam keadaan kosong."""
        self.lock = threading.Lock()
        
        # Dimulai dari None — belum ada data real. 
        # Akan diisi setelah frame pertama dari robot diproses.
        self._features: Optional[Dict[str, Any]] = None

    def update(self, new_features: Dict[str, Any]) -> None:
        """
        Memperbarui data fitur dengan hasil ekstraksi terbaru secara aman.

        Args:
            new_features (Dict[str, Any]): Dictionary fitur terbaru dari pipeline.
        """
        with self.lock:
            # Menyimpan salinan agar tidak terjadi mutasi referensi dari luar
            self._features = new_features.copy()

    def get(self) -> Optional[Dict[str, Any]]:
        """
        Mengambil salinan data fitur terbaru untuk dikirim sebagai JSON API.
        Mengembalikan None jika belum ada frame yang diproses.

        Returns:
            Optional[Dict[str, Any]]: Salinan dictionary fitur, atau None jika belum ada data.
        """
        with self.lock:
            if self._features is None:
                return None
            return self._features.copy()

    def has_data(self) -> bool:
        """Mengecek apakah sudah ada data fitur yang masuk."""
        with self.lock:
            return self._features is not None