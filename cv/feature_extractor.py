# cv/feature_extractor.py

from typing import Dict, Any, Optional

class FeatureExtractor:
    """
    Menyatukan seluruh hasil ekstraksi fitur menjadi satu payload dict.
    
    Kelas ini menyimpan state terakhir yang valid. Jika wajah tidak terdeteksi
    pada frame tertentu, kelas ini akan tetap mengembalikan struktur JSON yang utuh
    menggunakan nilai-nilai terakhir yang diketahui agar konsumen API tidak mengalami crash.
    """

    def __init__(self):
        """
        Inisialisasi state awal (default) untuk payload API.
        """
        self.last_state: Dict[str, Any] = {
            "ear": 0.0,
            "eye_status": "Open",
            "blink_count": 0,
            "blink_rate": 0.0,
            "eye_closure_duration": 0.0,
            "fps": 0.0,
            "timestamp": "",
            "face_detected": False
        }

    def build_payload(self, 
                      face_detected: bool, 
                      timestamp: str,
                      fps: float,
                      ear: Optional[float] = None, 
                      eye_status: Optional[str] = None, 
                      blink_count: Optional[int] = None, 
                      blink_rate: Optional[float] = None, 
                      closure_duration: Optional[float] = None) -> Dict[str, Any]:
        """
        Membangun payload dictionary sesuai dengan kontrak API akhir.

        Args:
            face_detected (bool): Apakah wajah ditemukan pada frame ini.
            timestamp (str): Waktu saat ini (ISO 8601).
            fps (float): Frame rate saat ini.
            ear (Optional[float]): Nilai EAR terbaru.
            eye_status (Optional[str]): Status mata terbaru.
            blink_count (Optional[int]): Total kedipan.
            blink_rate (Optional[float]): Laju kedipan per menit.
            closure_duration (Optional[float]): Durasi mata tertutup.

        Returns:
            Dict[str, Any]: Dictionary lengkap sesuai kontrak '/api/features'.
        """
        
        # Selalu perbarui flag deteksi, waktu, dan fps (karena ini independen dari wajah)
        self.last_state["face_detected"] = face_detected
        self.last_state["timestamp"] = timestamp
        self.last_state["fps"] = fps

        # Hanya perbarui metrik mata jika wajah terdeteksi
        if face_detected:
            if ear is not None:
                self.last_state["ear"] = ear
            if eye_status is not None:
                self.last_state["eye_status"] = eye_status
            if blink_count is not None:
                self.last_state["blink_count"] = blink_count
            if blink_rate is not None:
                self.last_state["blink_rate"] = blink_rate
            if closure_duration is not None:
                self.last_state["eye_closure_duration"] = closure_duration

        # Mengembalikan salinan (copy) dictionary untuk menghindari masalah mutasi 
        # referensi jika diakses oleh thread lain nantinya.
        return self.last_state.copy()