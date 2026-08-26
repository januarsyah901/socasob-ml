# cv/eye_analyzer.py
"""
eye_analyzer.py

High-level interface untuk analisis kondisi mata (kelelahan, mata kering, risiko miopia).
Mengintegrasikan modul scoring komposit dari eye_fatigue_scoring.py
(BlinkEventDetector, MetricsWindow, FatigueClassifier) menggantikan logika
single-threshold blink rate yang usang.
"""

from typing import Dict, Any, Optional, Tuple, List
from config import settings
from cv.eye_fatigue_scoring import (
    EyeState,
    SystemStatus,
    BlinkEventDetector,
    MetricsWindow,
    FatigueClassifier,
    compute_fatigue_score,
    calibrate_baseline
)
from utils.logger import get_logger

logger = get_logger(__name__)


class EyeConditionAnalyzer:
    """
    Orkestrator analisis kondisi mata pengguna:
    1. Mengubah nilai EAR menjadi event kedipan berdurasi (BlinkEventDetector).
    2. Menghitung metrik sliding window (MetricsWindow): smoothed rate, PERCLOS, duration, variability.
    3. Mengevaluasi skor komposit & status anti-flapping (FatigueClassifier).
    4. Menyediakan analisis risiko miopia dari jarak dan screen time.
    """

    def __init__(self,
                 ear_threshold: Optional[float] = None,
                 window_seconds: int = 60,
                 required_consecutive: int = 3,
                 min_data_quality: float = 0.7,
                 baseline_rate: float = 17.0):
        
        self.threshold = ear_threshold if ear_threshold is not None else getattr(settings, 'EAR_THRESHOLD', 0.23)
        self.baseline_rate = baseline_rate
        
        # Komponen scoring komposit
        self.detector = BlinkEventDetector(ear_threshold=self.threshold)
        self.window = MetricsWindow(window_seconds=window_seconds)
        self.classifier = FatigueClassifier(
            required_consecutive=required_consecutive,
            min_data_quality=min_data_quality
        )

        # Total akumulasi kedipan sejak session dimulai
        self.lifetime_blinks: int = 0

        # Batas jarak aman membaca (Widayat et al., 2026) dalam cm
        self.SAFE_DISTANCE_CM = 30
        # Batas screen time (Ha et al., 2025) dalam jam
        self.MAX_SCREEN_TIME_HOURS = 4

    def process_frame(self,
                      ear_value: float,
                      face_confidence: float,
                      timestamp: float) -> Tuple[bool, Dict[str, Any]]:
        """
        Memproses satu frame:
        - Catat frame ke sliding window
        - Update state machine kedipan
        - Evaluasi skor komposit dan klasifikasi fatigue

        Returns:
            Tuple[bool, Dict[str, Any]]: (is_blink_event, metrics_dict)
        """
        is_valid = face_confidence >= 0.5
        is_closed = (ear_value < self.detector.ear_threshold) if is_valid else False

        # 1. Update sliding window frame log
        self.window.add_frame(timestamp, is_closed=is_closed, is_valid=is_valid)

        # 2. Update detector kedipan
        event = self.detector.update(ear_value, face_confidence, timestamp)
        blink_event = False
        if event:
            blink_event = True
            self.lifetime_blinks += 1
            self.window.add_blink(event)

        # 3. Evaluasi klasifikasi fatigue
        candidate_status, stable_status, detail = self.classifier.evaluate(
            self.window, baseline_rate=self.baseline_rate
        )

        # 4. Petakan stable_status ke health_status, conditions, dan recommendations
        status_str, conditions, recommendations = self._map_status_details(stable_status, detail)

        metrics_dict = {
            "eye_state": self.detector.state.value,
            "blink_event": blink_event,
            "blink_count": len(self.window.blink_events),
            "lifetime_blinks": self.lifetime_blinks,
            "raw_blink_rate": round(self.window.raw_blink_rate_per_minute(), 2),
            "smoothed_blink_rate": round(detail["rate"], 2) if detail else round(self.window.smoothed_blink_rate(), 2),
            "perclos": round(detail["perclos"], 3) if detail else round(self.window.perclos(), 3),
            "avg_blink_duration": round(detail["avg_duration"], 3) if detail else round(self.window.avg_blink_duration(), 3),
            "interval_variability": round(detail["interval_variability"], 3) if detail else round(self.window.interval_variability(), 3),
            "composite_score": round(detail["composite_score"], 1) if detail else 0.0,
            "data_quality": round(self.window.data_quality(), 2),
            "candidate_status": candidate_status.value,
            "system_status": stable_status.value,
            "health_status": status_str,
            "conditions": conditions,
            "recommendations": recommendations,
        }

        return blink_event, metrics_dict

    def _map_status_details(self,
                            status: SystemStatus,
                            detail: Optional[Dict[str, float]]) -> Tuple[str, List[str], List[str]]:
        """
        Menghasilkan label kondisi dan rekomendasi medis berdasarkan status sistem.
        """
        if status == SystemStatus.AMAN:
            return (
                "Aman",
                ["Normal"],
                ["Pertahankan frekuensi kedipan dan istirahat teratur."]
            )
        elif status == SystemStatus.PERINGATAN_RINGAN:
            return (
                "Peringatan",
                ["Risiko Mata Kering (Instabilitas Tear Film)"],
                ["Tingkatkan frekuensi kedipan Anda secara sadar untuk melembapkan kornea."]
            )
        elif status == SystemStatus.PERINGATAN_BERAT:
            return (
                "Peringatan",
                ["Mata Lelah (Astenopia) Berat & Mata Kering Kritis"],
                ["Segera lakukan aturan 20-20-20: Istirahat 20 detik melihat jarak 20 kaki (6 meter)."]
            )
        else:  # NO_DATA
            return (
                "Data Tidak Valid",
                ["Kualitas Landmark Wajah Rendah"],
                ["Posisikan wajah menghadap kamera dengan pencahayaan yang cukup."]
            )

    def analyze_myopia_risk(self, distance_cm: Optional[float] = None, screen_time_hours: Optional[float] = None) -> Dict[str, Any]:
        """
        Menganalisis risiko Miopia berdasarkan jarak pandang dan waktu layar.
        """
        risk_level = "Rendah"
        factors = []

        if distance_cm is not None and distance_cm < self.SAFE_DISTANCE_CM:
            risk_level = "Tinggi"
            factors.append(f"Jarak pandang terlalu dekat ({distance_cm} cm).")

        if screen_time_hours is not None and screen_time_hours > self.MAX_SCREEN_TIME_HOURS:
            risk_level = "Sangat Tinggi"
            factors.append(f"Screen time berlebih (>4 jam/hari). Risiko 97% lebih tinggi.")

        return {
            "myopia_risk": risk_level,
            "risk_factors": factors
        }