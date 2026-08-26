# scoring/fatigue_score.py
"""
fatigue_score.py — Modul A: Scoring komposit kelelahan & risiko mata kering.

Pipeline:
    MetricsWindow → compute_fatigue_score() → FatigueClassifier (hysteresis)
    → Status resmi (Aman / Peringatan Ringan / Peringatan Berat / No Data)

Composite score (0–100), bobot:
    PERCLOS          35%  — paling diagnostik, langsung mencerminkan penutupan kelopak
    Blink rate dev.  30%  — deviasi dari baseline personal
    Durasi kedipan   20%  — kedipan memanjang (>0.4s) = kelelahan otot kelopak
    Variabilitas     15%  — pola tidak teratur = degradasi arousal

Tabel klasifikasi (tervalidasi Chai dkk., 2025, Scientific Reports, n=45,
dewasa muda bergejala mata kering):
    Skor < 30    → Aman
    Skor 30–59   → Peringatan Ringan
    Skor ≥ 60    → Peringatan Berat
    data_quality < 70% → No Data

DISCLAIMER: Ambang dan bobot ini berdasarkan studi Chai dkk. (2025) pada
populasi spesifik (n=45, dewasa muda bergejala mata kering). Generalisasi
ke populasi lain memerlukan validasi tambahan.

Data acuan blink rate:
- Kedipan spontan rata-rata 27.75 ± 14.43/menit berkorelasi dengan
  tear meniscus height (r=0.48; p<0.01).
- 10 kedipan/menit signifikan memperburuk TMH, BUT, bulbar redness,
  SANDE (p<0.0002 s.d. p<0.0001) dibanding spontan.
- Tidak ada beda signifikan antara spontan dan 20 kedipan/menit.
- → Ambang aman minimal ≥20 kedipan/menit.
"""

from enum import Enum
from typing import Dict, Optional, Tuple

from vision.metrics_window import MetricsWindow


class SystemStatus(Enum):
    """Status sistem setelah evaluasi hysteresis."""
    AMAN = "Aman"
    PERINGATAN_RINGAN = "Peringatan - Risiko Mata Kering"
    PERINGATAN_BERAT = "Peringatan - Mata Lelah Berat / Mata Kering Kritis"
    NO_DATA = "Data Tidak Valid"


# ─────────────────────────────────────────────
# Composite Scoring
# ─────────────────────────────────────────────

# Bobot fitur — PERCLOS & durasi diberi bobot lebih besar karena
# lebih diagnostik secara klinis (Chai dkk., 2025).
WEIGHTS = {
    "perclos": 0.35,
    "rate": 0.30,
    "duration": 0.20,
    "variability": 0.15,
}


def compute_fatigue_score(
    metrics: MetricsWindow,
    baseline_rate: float = 17.0,
) -> Tuple[float, Dict[str, float]]:
    """
    Hitung skor komposit 0–100 (makin tinggi = makin berisiko).

    Args:
        metrics: Sliding window berisi data kedipan & frame.
        baseline_rate: Baseline personal blink rate (kedipan/menit).
                       Default 17.0 = rata-rata populasi (fallback).

    Returns:
        Tuple (composite_score, detail_dict).
    """
    rate = metrics.smoothed_blink_rate()
    perclos = metrics.perclos()
    avg_dur = metrics.avg_blink_duration()
    iv = metrics.interval_variability()

    # Sub-score masing-masing fitur (dinormalisasi ke 0–100)

    # Rate: deviasi dari baseline personal, dinormalisasi
    rate_deviation = abs(rate - baseline_rate) / max(baseline_rate, 1.0)
    rate_score = min(rate_deviation, 1.0) * 100

    # PERCLOS: >0.15 (15%) umum dipakai literatur sebagai ambang fatigue tinggi
    perclos_score = min(perclos / 0.15, 1.0) * 100

    # Durasi kedipan: memanjang (>0.4 detik) menandakan kelelahan otot kelopak
    # Hanya aktif jika durasi > 0.2s (batas bawah kedipan normal)
    duration_score = min(avg_dur / 0.4, 1.0) * 100 if avg_dur > 0.2 else 0.0

    # Variabilitas interval: CV > 1.0 = sangat tidak teratur
    variability_score = min(iv, 1.0) * 100

    composite = (
        WEIGHTS["perclos"] * perclos_score
        + WEIGHTS["rate"] * rate_score
        + WEIGHTS["duration"] * duration_score
        + WEIGHTS["variability"] * variability_score
    )

    detail = {
        "rate": rate,
        "perclos": perclos,
        "avg_duration": avg_dur,
        "interval_variability": iv,
        "incomplete_blink_ratio": metrics.incomplete_blink_ratio(),
        "composite_score": composite,
        "rate_score": rate_score,
        "perclos_score": perclos_score,
        "duration_score": duration_score,
        "variability_score": variability_score,
    }
    return composite, detail


# ─────────────────────────────────────────────
# Hysteresis State Machine (anti-flapping / debounce)
# ─────────────────────────────────────────────

class FatigueClassifier:
    """
    Mengubah composite score menjadi status resmi dengan hysteresis:
    status baru hanya berlaku setelah N evaluasi berturut-turut menunjukkan
    status yang sama (debounce), mencegah flip-flop akibat noise sesaat.

    Prinsip: Hysteresis, bukan snapshot instan (Bagian 3).
    """

    # Ambang klasifikasi (Chai dkk., 2025)
    THRESHOLD_AMAN: float = 30.0
    THRESHOLD_RINGAN: float = 60.0  # ≥60 = Peringatan Berat

    def __init__(
        self,
        required_consecutive: int = 3,
        min_data_quality: float = 0.70,
    ):
        """
        Args:
            required_consecutive: Jumlah evaluasi berturut-turut yang harus
                                  menunjukkan status baru sebelum status resmi berubah.
            min_data_quality: Minimum data quality untuk evaluasi valid.
                              Di bawah ini → status NO_DATA.
        """
        self.required_consecutive = required_consecutive
        self.min_data_quality = min_data_quality

        self.current_status = SystemStatus.AMAN
        self._pending_status: Optional[SystemStatus] = None
        self._pending_count: int = 0

    def _score_to_status(self, score: float) -> SystemStatus:
        if score < self.THRESHOLD_AMAN:
            return SystemStatus.AMAN
        elif score < self.THRESHOLD_RINGAN:
            return SystemStatus.PERINGATAN_RINGAN
        else:
            return SystemStatus.PERINGATAN_BERAT

    def evaluate(
        self,
        metrics: MetricsWindow,
        baseline_rate: float = 17.0,
    ) -> Tuple[SystemStatus, SystemStatus, Optional[Dict[str, float]]]:
        """
        Evaluasi satu siklus: hitung score, tentukan candidate, terapkan hysteresis.

        Args:
            metrics: Sliding window dengan data kedipan.
            baseline_rate: Baseline blink rate personal.

        Returns:
            Tuple (candidate_status, stable_status, detail_or_None).
            - candidate_status: status yang diusulkan evaluasi saat ini.
            - stable_status: status resmi setelah hysteresis (yang dikirim ke klien).
            - detail: dict metrik detail, atau None jika data quality rendah.
        """
        quality = metrics.data_quality()
        if quality < self.min_data_quality:
            return SystemStatus.NO_DATA, self.current_status, None

        score, detail = compute_fatigue_score(metrics, baseline_rate)
        candidate = self._score_to_status(score)

        if candidate == self.current_status:
            # Status sama → reset pending
            self._pending_status = None
            self._pending_count = 0
        else:
            if candidate == self._pending_status:
                self._pending_count += 1
            else:
                self._pending_status = candidate
                self._pending_count = 1

            # Promosikan ke status resmi jika konsisten N kali berturut-turut
            if self._pending_count >= self.required_consecutive:
                self.current_status = candidate
                self._pending_status = None
                self._pending_count = 0

        return candidate, self.current_status, detail


# ─────────────────────────────────────────────
# Kalibrasi Baseline Personal
# ─────────────────────────────────────────────

def calibrate_baseline(
    blink_timestamps: list,
    calibration_seconds: int = 90,
) -> float:
    """
    Hitung baseline blink rate personal dari sesi kalibrasi singkat.

    User diminta duduk santai / baca teks netral selama ~90 detik.
    Jika tidak dilakukan, fallback ke rata-rata populasi 17 bpm.

    WARNING: 17 bpm adalah rata-rata populasi umum. Nilai personal
    bisa sangat berbeda (studi menunjukkan range 10–30+ bpm untuk
    individu sehat). Gunakan kalibrasi untuk akurasi terbaik.

    Args:
        blink_timestamps: List timestamp (detik) setiap kedipan selama kalibrasi.
        calibration_seconds: Durasi sesi kalibrasi (default 90 detik).

    Returns:
        Baseline blink rate dalam kedipan/menit.
    """
    if not blink_timestamps or calibration_seconds <= 0:
        return 17.0  # fallback rata-rata populasi
    n = len(blink_timestamps)
    return (n / calibration_seconds) * 60.0
