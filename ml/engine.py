# ml/engine.py
"""
engine.py — ML / Inference Engine.

Lapisan ini menerima fitur mentah dari ketiga modul deteksi dan
menghasilkan tiga keluaran matang sebagai komponen independen:

1. FatigueDetector — konsumsi composite score & hysteresis dari Modul A.
2. DryEyeDetector — sub-klasifikasi PERCLOS, durasi kedip, variabilitas
                    dari Modul A (terpisah dari fatigue murni, walau
                    sumber fiturnya sama).
3. MyopiaRiskModel — gabungan peringatan instan Modul B1 dengan
                     estimasi persentase Modul B2.

CATATAN IMPLEMENTASI:
Ketiga komponen ini saat ini berupa rule-based scoring sesuai
spesifikasi Bagian 4. Desain interface menggunakan Protocol (structural
subtyping) agar rule-based engine ini bisa diganti dengan model terlatih
di masa depan tanpa mengubah kontrak data ke lapisan Database/WebSocket.
"""

import time
from typing import Dict, Any, Optional, Protocol, runtime_checkable

from scoring.fatigue_score import (
    FatigueClassifier,
    SystemStatus,
    compute_fatigue_score,
)
from scoring.active_myopia_guard import ActiveMyopiaGuard
from scoring.myopia_risk import MyopiaRiskEstimator
from vision.metrics_window import MetricsWindow


# ─────────────────────────────────────────────
# Protocol (interface) untuk detector yang bisa diganti
# ─────────────────────────────────────────────

@runtime_checkable
class BaseDetector(Protocol):
    """
    Protocol untuk semua detector.

    Rule-based atau model terlatih harus mengimplementasikan
    method predict() dengan signature yang sama.
    Menggunakan Protocol (bukan ABC) agar mendukung structural subtyping
    — class tidak perlu inherit, cukup punya method yang cocok.
    """

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Jalankan inferensi dan kembalikan hasil prediksi.

        Args:
            features: Dict fitur mentah dari modul deteksi.

        Returns:
            Dict hasil prediksi dengan field 'status', 'detail', dll.
        """
        ...


# ─────────────────────────────────────────────
# 1. Fatigue Detector (Rule-Based)
# ─────────────────────────────────────────────

class RuleBasedFatigueDetector:
    """
    Detector kelelahan mata berbasis aturan.

    Mengonsumsi composite score dan hysteresis state machine dari Modul A
    (scoring/fatigue_score.py). Menghasilkan status dan rekomendasi.
    """

    STATUS_RECOMMENDATIONS = {
        SystemStatus.AMAN: {
            "label": "Aman",
            "recommendation": "Pertahankan pola kedipan & aktivitas saat ini.",
        },
        SystemStatus.PERINGATAN_RINGAN: {
            "label": "Peringatan Ringan",
            "recommendation": (
                "Tingkatkan frekuensi kedipan secara sadar; "
                "jaga kelembapan ruangan."
            ),
        },
        SystemStatus.PERINGATAN_BERAT: {
            "label": "Peringatan Berat",
            "recommendation": "Terapkan aturan 20-20-20.",
        },
        SystemStatus.NO_DATA: {
            "label": "No Data",
            "recommendation": (
                "Posisikan wajah menghadap kamera dengan pencahayaan cukup."
            ),
        },
    }

    def __init__(self, baseline_rate: float = 17.0):
        self.baseline_rate = baseline_rate
        self.classifier = FatigueClassifier(
            required_consecutive=3,
            min_data_quality=0.70,
        )

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Args:
            features: Harus berisi key 'metrics_window' (MetricsWindow instance).

        Returns:
            Dict dengan: type, status, composite_score, detail, recommendation, timestamp.
        """
        metrics: MetricsWindow = features["metrics_window"]
        candidate, stable, detail = self.classifier.evaluate(
            metrics, baseline_rate=self.baseline_rate,
        )

        rec = self.STATUS_RECOMMENDATIONS.get(
            stable, self.STATUS_RECOMMENDATIONS[SystemStatus.NO_DATA]
        )

        return {
            "type": "fatigue",
            "status": stable.value,
            "candidate_status": candidate.value,
            "composite_score": round(detail["composite_score"], 1) if detail else 0.0,
            "data_quality": round(metrics.data_quality(), 2),
            "detail": detail,
            "recommendation": rec["recommendation"],
            "label": rec["label"],
            "timestamp": time.time(),
        }


# ─────────────────────────────────────────────
# 2. Dry Eye Detector (Rule-Based)
# ─────────────────────────────────────────────

class RuleBasedDryEyeDetector:
    """
    Detector risiko mata kering berbasis aturan.

    Sub-klasifikasi terpisah dari fatigue murni — meskipun sumber fiturnya
    sama (PERCLOS, durasi kedip, variabilitas dari Modul A).

    Fitur kunci untuk dry eye (berbeda penekanan dari fatigue):
    - Incomplete blink ratio > 0.5 → risiko signifikan
    - PERCLOS rendah + blink rate rendah → mata jarang menutup sempurna
    - Variabilitas interval tinggi → pola kedipan tidak teratur
    """

    # Ambang khusus dry eye
    INCOMPLETE_RATIO_WARN: float = 0.3
    INCOMPLETE_RATIO_HIGH: float = 0.5
    LOW_BLINK_RATE: float = 10.0  # bpm — signifikan perburuk TMH, BUT, dll.

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        metrics: MetricsWindow = features["metrics_window"]

        quality = metrics.data_quality()
        if quality < 0.70:
            return {
                "type": "dry_eye",
                "status": "No Data",
                "perclos": 0.0,
                "avg_blink_duration": 0.0,
                "incomplete_blink_ratio": 0.0,
                "recommendation": "Data tidak cukup untuk evaluasi.",
                "timestamp": time.time(),
            }

        perclos = metrics.perclos()
        avg_dur = metrics.avg_blink_duration()
        iv = metrics.interval_variability()
        incomplete_ratio = metrics.incomplete_blink_ratio()
        rate = metrics.smoothed_blink_rate()

        # Scoring risiko dry eye
        risk_factors = []
        status = "Aman"

        # Incomplete blinks — klinis paling relevan untuk dry eye
        if incomplete_ratio >= self.INCOMPLETE_RATIO_HIGH:
            risk_factors.append(
                f"Proporsi kedipan tidak sempurna sangat tinggi ({incomplete_ratio:.0%})"
            )
            status = "Peringatan Berat"
        elif incomplete_ratio >= self.INCOMPLETE_RATIO_WARN:
            risk_factors.append(
                f"Proporsi kedipan tidak sempurna meningkat ({incomplete_ratio:.0%})"
            )
            if status == "Aman":
                status = "Peringatan Ringan"

        # Blink rate rendah
        # 10 bpm signifikan memperburuk TMH, BUT, bulbar redness, SANDE
        # (p<0.0002 s.d. p<0.0001) dibanding spontan.
        if rate < self.LOW_BLINK_RATE:
            risk_factors.append(
                f"Blink rate rendah ({rate:.1f}/menit < 10/menit)"
            )
            status = "Peringatan Berat"

        # Durasi kedipan memanjang + variabilitas tinggi
        if avg_dur > 0.4 and iv > 0.5:
            risk_factors.append("Pola kedipan tidak teratur dengan durasi memanjang")
            if status == "Aman":
                status = "Peringatan Ringan"

        recommendation = "Pertahankan pola kedipan saat ini."
        if status == "Peringatan Ringan":
            recommendation = (
                "Tingkatkan frekuensi kedipan sadar; gunakan tetes mata "
                "pelembap jika perlu."
            )
        elif status == "Peringatan Berat":
            recommendation = (
                "Segera istirahat. Lakukan kedipan sadar yang sempurna "
                "(tutup mata sepenuhnya). Pertimbangkan tetes mata pelembap."
            )

        return {
            "type": "dry_eye",
            "status": status,
            "perclos": round(perclos, 3),
            "avg_blink_duration": round(avg_dur, 3),
            "incomplete_blink_ratio": round(incomplete_ratio, 3),
            "blink_rate": round(rate, 1),
            "interval_variability": round(iv, 3),
            "risk_factors": risk_factors,
            "recommendation": recommendation,
            "timestamp": time.time(),
        }


# ─────────────────────────────────────────────
# 3. Myopia Risk Model (Rule-Based)
# ─────────────────────────────────────────────

class RuleBasedMyopiaRiskModel:
    """
    Gabungan Modul B1 (peringatan instan jarak & durasi) dengan
    Modul B2 (estimasi risiko kumulatif dari screen time).
    """

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Args:
            features: Harus berisi:
                - 'guard_result': Dict output dari ActiveMyopiaGuard.update()
                - 'risk_result': Dict output dari MyopiaRiskEstimator.get_risk()
        """
        guard: Dict = features.get("guard_result", {})
        risk: Dict = features.get("risk_result", {})

        distance_warning = guard.get("distance_warning", False)
        break_state = guard.get("break_state", "active")
        risk_pct = risk.get("risk_percentage", 0.0)
        screen_time_min = risk.get("screen_time_minutes", 0.0)

        # Status gabungan
        warnings = []
        status = "Aman"

        if distance_warning:
            warnings.append(
                f"Jarak terlalu dekat ({guard.get('distance_cm', '?')} cm < 50 cm)"
            )
            status = "Peringatan"

        if break_state == "break_needed":
            warnings.append("Waktu istirahat! Lihat objek sejauh 6 meter selama 20 detik.")
            status = "Peringatan"

        if risk_pct >= 97:
            warnings.append(
                f"Screen time tinggi ({screen_time_min:.0f} menit). "
                f"Peningkatan risiko miopia: +{risk_pct}%."
            )
            status = "Peringatan Berat"
        elif risk_pct >= 47:
            warnings.append(
                f"Screen time sedang ({screen_time_min:.0f} menit). "
                f"Peningkatan risiko miopia: +{risk_pct}%."
            )
            if status == "Aman":
                status = "Peringatan Ringan"

        recommendation = "Pertahankan jarak dan durasi layar saat ini."
        if status != "Aman":
            recommendation = (
                "Jaga jarak ≥50 cm dari layar. Terapkan aturan 20-20-20. "
                "Batasi screen time harian."
            )

        return {
            "type": "myopia_risk",
            "status": status,
            "distance_cm": guard.get("distance_cm"),
            "distance_warning": distance_warning,
            "break_state": break_state,
            "work_elapsed_sec": guard.get("work_elapsed_sec", 0),
            "break_remaining_sec": guard.get("break_remaining_sec", 0),
            "screen_time_minutes": round(screen_time_min, 1),
            "risk_percentage": risk_pct,
            "odds_ratio": risk.get("odds_ratio", 1.0),
            "distance_warning_count": guard.get("distance_warning_count", 0),
            "break_reminder_count": guard.get("break_reminder_count", 0),
            "warnings": warnings,
            "recommendation": recommendation,
            "timestamp": time.time(),
        }


# ─────────────────────────────────────────────
# Inference Engine (orkestrator)
# ─────────────────────────────────────────────

class InferenceEngine:
    """
    Orkestrator inference yang menjalankan ketiga detector dan
    menghasilkan hasil gabungan untuk lapisan Database/WebSocket.

    Setiap detector independen — bisa di-update/dilatih ulang terpisah.
    Untuk mengganti rule-based dengan model terlatih, cukup ganti instance
    detector yang sesuai (selama method predict() mengikuti Protocol).
    """

    def __init__(
        self,
        fatigue_detector: Optional[RuleBasedFatigueDetector] = None,
        dry_eye_detector: Optional[RuleBasedDryEyeDetector] = None,
        myopia_risk_model: Optional[RuleBasedMyopiaRiskModel] = None,
        baseline_rate: float = 17.0,
    ):
        self.fatigue = fatigue_detector or RuleBasedFatigueDetector(
            baseline_rate=baseline_rate
        )
        self.dry_eye = dry_eye_detector or RuleBasedDryEyeDetector()
        self.myopia = myopia_risk_model or RuleBasedMyopiaRiskModel()

    def run(
        self,
        metrics_window: MetricsWindow,
        guard_result: Dict[str, Any],
        risk_result: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Jalankan semua detector dan kembalikan hasil gabungan.

        Args:
            metrics_window: Sliding window dari vision pipeline.
            guard_result: Output ActiveMyopiaGuard.update().
            risk_result: Output MyopiaRiskEstimator.get_risk().

        Returns:
            Dict dengan key 'fatigue', 'dry_eye', 'myopia_risk'.
            Masing-masing berisi dict hasil prediksi detector.
        """
        fatigue_result = self.fatigue.predict({"metrics_window": metrics_window})
        dry_eye_result = self.dry_eye.predict({"metrics_window": metrics_window})
        myopia_result = self.myopia.predict({
            "guard_result": guard_result,
            "risk_result": risk_result,
        })

        return {
            "fatigue": fatigue_result,
            "dry_eye": dry_eye_result,
            "myopia_risk": myopia_result,
        }

    def update_baseline(self, new_baseline: float) -> None:
        """Update baseline blink rate setelah kalibrasi personal."""
        if isinstance(self.fatigue, RuleBasedFatigueDetector):
            self.fatigue.baseline_rate = new_baseline
