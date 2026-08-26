# cv/feature_extractor.py

from typing import Dict, Any, Optional

class FeatureExtractor:
    """
    Menyatukan seluruh hasil ekstraksi fitur menjadi satu payload dict yang komprehensif.
    
    Menyimpan state terakhir yang valid agar jika wajah/landmark sempat hilang sesaat,
    struktur JSON tetap konsisten dan aman untuk API konsumen.
    """

    def __init__(self):
        """
        Inisialisasi state default untuk payload API.
        """
        self.last_state: Dict[str, Any] = {
            "ear": 0.0,
            "eye_status": "Open",
            "blink_count": 0,
            "lifetime_blinks": 0,
            "blink_rate": 0.0,
            "raw_blink_rate": 0.0,
            "eye_closure_duration": 0.0,
            "perclos": 0.0,
            "composite_score": 0.0,
            "avg_blink_duration": 0.0,
            "interval_variability": 0.0,
            "data_quality": 1.0,
            "system_status": "Aman",
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
                      lifetime_blinks: Optional[int] = None,
                      blink_rate: Optional[float] = None, 
                      raw_blink_rate: Optional[float] = None,
                      closure_duration: Optional[float] = None,
                      perclos: Optional[float] = None,
                      composite_score: Optional[float] = None,
                      avg_blink_duration: Optional[float] = None,
                      interval_variability: Optional[float] = None,
                      data_quality: Optional[float] = None,
                      system_status: Optional[str] = None) -> Dict[str, Any]:
        """
        Membangun payload dictionary lengkap sesuai dengan kontrak '/api/features'.
        """
        self.last_state["face_detected"] = face_detected
        self.last_state["timestamp"] = timestamp
        self.last_state["fps"] = fps

        if face_detected:
            if ear is not None:
                self.last_state["ear"] = ear
            if eye_status is not None:
                self.last_state["eye_status"] = eye_status
            if blink_count is not None:
                self.last_state["blink_count"] = blink_count
            if lifetime_blinks is not None:
                self.last_state["lifetime_blinks"] = lifetime_blinks
            if blink_rate is not None:
                self.last_state["blink_rate"] = blink_rate
            if raw_blink_rate is not None:
                self.last_state["raw_blink_rate"] = raw_blink_rate
            if closure_duration is not None:
                self.last_state["eye_closure_duration"] = closure_duration
            if perclos is not None:
                self.last_state["perclos"] = perclos
            if composite_score is not None:
                self.last_state["composite_score"] = composite_score
            if avg_blink_duration is not None:
                self.last_state["avg_blink_duration"] = avg_blink_duration
            if interval_variability is not None:
                self.last_state["interval_variability"] = interval_variability
            if data_quality is not None:
                self.last_state["data_quality"] = data_quality
            if system_status is not None:
                self.last_state["system_status"] = system_status

        return self.last_state.copy()