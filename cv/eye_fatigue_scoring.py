# cv/eye_fatigue_scoring.py
"""
eye_fatigue_scoring.py

Modul scoring untuk deteksi mata lelah & mata kering berbasis kombinasi
fitur kedipan (blink rate, durasi kedip, PERCLOS, variabilitas interval),
dilengkapi smoothing, confidence-gating, dan hysteresis state machine
supaya tidak salah trigger dari noise sesaat / kegagalan deteksi landmark.
"""

import time
import numpy as np
from collections import deque
from enum import Enum
from typing import Optional, Dict, Any, Tuple


class EyeState(Enum):
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"


class SystemStatus(Enum):
    AMAN = "Aman"
    PERINGATAN_RINGAN = "Peringatan - Risiko Mata Kering"
    PERINGATAN_BERAT = "Peringatan - Mata Lelah Berat / Mata Kering Kritis"
    NO_DATA = "Data Tidak Valid"


# ---------------------------------------------------------------------
# 1. Blink Event Detector (dari EAR / MediaPipe)
# ---------------------------------------------------------------------
class BlinkEventDetector:
    """
    Mengubah aliran nilai EAR (Eye Aspect Ratio) per-frame menjadi
    event kedipan diskrit lengkap dengan durasi kedip.
    Frame dengan confidence landmark rendah diabaikan (bukan dianggap "tidak berkedip").
    """

    def __init__(self, ear_threshold=0.21, min_closed_frames=2, fps=30):
        self.ear_threshold = ear_threshold
        self.min_closed_frames = min_closed_frames
        self.fps = fps

        self.state = EyeState.OPEN
        self.closed_frame_count = 0
        self.blink_start_time = None

    def update(self, ear_value: float, face_confidence: float, timestamp: float) -> Optional[Dict[str, float]]:
        """
        Return: dict blink_event {"duration": float detik, "timestamp": float} atau None
        """
        if face_confidence < 0.5:
            return None  # data quality gate: jangan proses frame yang tidak andal

        if ear_value < self.ear_threshold:
            if self.state == EyeState.OPEN:
                self.state = EyeState.CLOSING
                self.blink_start_time = timestamp
                self.closed_frame_count = 1
            elif self.state == EyeState.CLOSING:
                self.closed_frame_count += 1
                if self.closed_frame_count >= self.min_closed_frames:
                    self.state = EyeState.CLOSED
        else:
            if self.state in (EyeState.CLOSING, EyeState.CLOSED):
                duration = timestamp - (self.blink_start_time if self.blink_start_time else timestamp)
                event = {"duration": duration, "timestamp": timestamp}
                self.state = EyeState.OPEN
                self.closed_frame_count = 0
                self.blink_start_time = None
                return event
            self.state = EyeState.OPEN

        return None


# ---------------------------------------------------------------------
# 2. Sliding Window Metrics (rate, PERCLOS, variabilitas)
# ---------------------------------------------------------------------
class MetricsWindow:
    """
    Menyimpan histori kedipan & frame dalam window waktu tertentu,
    menghitung fitur-fitur turunan dengan smoothing (bukan angka instan).
    """

    def __init__(self, window_seconds=60, smoothing_alpha=0.3):
        self.window_seconds = window_seconds
        self.smoothing_alpha = smoothing_alpha

        self.blink_events = deque()      # (timestamp, duration)
        self.closed_frame_log = deque()  # (timestamp, is_closed)
        self.valid_frame_log = deque()   # (timestamp, is_valid)

        self._smoothed_rate = None

    def add_blink(self, event: Dict[str, float]):
        self.blink_events.append((event["timestamp"], event["duration"]))
        self._trim()

    def add_frame(self, timestamp: float, is_closed: bool, is_valid: bool):
        self.closed_frame_log.append((timestamp, is_closed))
        self.valid_frame_log.append((timestamp, is_valid))
        self._trim()

    def _trim(self):
        cutoff = time.time() - self.window_seconds
        for log in (self.blink_events, self.closed_frame_log, self.valid_frame_log):
            while log and log[0][0] < cutoff:
                log.popleft()

    def data_quality(self) -> float:
        """Proporsi frame valid (landmark terdeteksi baik) dalam window."""
        if not self.valid_frame_log:
            return 0.0
        valid = sum(1 for _, v in self.valid_frame_log if v)
        return valid / len(self.valid_frame_log)

    def raw_blink_rate_per_minute(self) -> float:
        n = len(self.blink_events)
        elapsed = max(self.window_seconds, 1)
        return (n / elapsed) * 60.0

    def smoothed_blink_rate(self) -> float:
        raw = self.raw_blink_rate_per_minute()
        if self._smoothed_rate is None:
            self._smoothed_rate = raw
        else:
            a = self.smoothing_alpha
            self._smoothed_rate = a * raw + (1 - a) * self._smoothed_rate
        return self._smoothed_rate

    def perclos(self) -> float:
        """Percentage of eyelid closure over the window."""
        if not self.closed_frame_log:
            return 0.0
        closed = sum(1 for _, c in self.closed_frame_log if c)
        return closed / len(self.closed_frame_log)

    def avg_blink_duration(self) -> float:
        if not self.blink_events:
            return 0.0
        durations = [d for _, d in self.blink_events]
        return float(np.mean(durations))

    def interval_variability(self) -> float:
        """Coefficient of variation dari interval antar-kedipan (std/mean)."""
        timestamps = [t for t, _ in self.blink_events]
        if len(timestamps) < 3:
            return 0.0
        intervals = np.diff(timestamps)
        mean_iv = np.mean(intervals)
        if mean_iv == 0:
            return 0.0
        return float(np.std(intervals) / mean_iv)


# ---------------------------------------------------------------------
# 3. Composite Scoring (bukan single-threshold pada rate)
# ---------------------------------------------------------------------
def compute_fatigue_score(metrics: MetricsWindow, baseline_rate=17.0) -> Tuple[float, Dict[str, float]]:
    """
    Skor komposit 0-100 (makin tinggi = makin berisiko), gabungan dari
    beberapa fitur agar tidak salah simpul hanya dari blink rate.
    """
    rate = metrics.smoothed_blink_rate()
    perclos = metrics.perclos()
    avg_dur = metrics.avg_blink_duration()
    iv = metrics.interval_variability()

    rate_deviation = abs(rate - baseline_rate) / baseline_rate
    rate_score = min(rate_deviation, 1.0) * 100

    # >0.15 PERCLOS umum dipakai literatur sebagai ambang fatigue tinggi
    perclos_score = min(perclos / 0.15, 1.0) * 100

    # durasi kedip memanjang (>0.4 detik) menandakan kelelahan otot kelopak
    duration_score = min(avg_dur / 0.4, 1.0) * 100 if avg_dur > 0.2 else 0

    variability_score = min(iv, 1.0) * 100

    # PERCLOS & durasi kedip diberi bobot lebih besar krn lebih diagnostik
    weights = {"rate": 0.30, "perclos": 0.35, "duration": 0.20, "variability": 0.15}

    composite = (
        weights["rate"] * rate_score
        + weights["perclos"] * perclos_score
        + weights["duration"] * duration_score
        + weights["variability"] * variability_score
    )
    return composite, {
        "rate": rate,
        "perclos": perclos,
        "avg_duration": avg_dur,
        "interval_variability": iv,
        "composite_score": composite,
    }


# ---------------------------------------------------------------------
# 4. State Machine dengan Hysteresis (anti-flapping)
# ---------------------------------------------------------------------
class FatigueClassifier:
    """
    Skor komposit -> status sistem, dengan syarat konsistensi N evaluasi
    berturut-turut sebelum status resmi berubah (debounce).
    """

    THRESHOLDS = {"aman": 30, "peringatan_ringan": 60}  # >60 -> peringatan_berat

    def __init__(self, required_consecutive=3, min_data_quality=0.7):
        self.required_consecutive = required_consecutive
        self.min_data_quality = min_data_quality

        self.current_status = SystemStatus.AMAN
        self.pending_status = None
        self.pending_count = 0

    def _score_to_status(self, score: float) -> SystemStatus:
        if score < self.THRESHOLDS["aman"]:
            return SystemStatus.AMAN
        elif score < self.THRESHOLDS["peringatan_ringan"]:
            return SystemStatus.PERINGATAN_RINGAN
        else:
            return SystemStatus.PERINGATAN_BERAT

    def evaluate(self, metrics: MetricsWindow, baseline_rate=17.0) -> Tuple[SystemStatus, SystemStatus, Optional[Dict[str, float]]]:
        quality = metrics.data_quality()
        if quality < self.min_data_quality:
            # deteksi tidak andal -> jangan ubah status resmi, laporkan NO_DATA saja
            return SystemStatus.NO_DATA, self.current_status, None

        score, detail = compute_fatigue_score(metrics, baseline_rate)
        candidate = self._score_to_status(score)

        if candidate == self.current_status:
            self.pending_status = None
            self.pending_count = 0
        else:
            if candidate == self.pending_status:
                self.pending_count += 1
            else:
                self.pending_status = candidate
                self.pending_count = 1

            if self.pending_count >= self.required_consecutive:
                self.current_status = candidate
                self.pending_status = None
                self.pending_count = 0

        return candidate, self.current_status, detail


# ---------------------------------------------------------------------
# 5. Kalibrasi Baseline Personal (opsional, dijalankan di awal sesi)
# ---------------------------------------------------------------------
def calibrate_baseline(blink_timestamps: list, calibration_seconds=90) -> float:
    """
    Hitung baseline blink rate personal dari sesi kalibrasi singkat
    (misal user diminta duduk santai/baca teks netral selama ~90 detik).
    """
    if not blink_timestamps or calibration_seconds <= 0:
        return 17.0  # fallback ke rata-rata populasi
    n = len(blink_timestamps)
    return (n / calibration_seconds) * 60.0
