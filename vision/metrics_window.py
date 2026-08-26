# vision/metrics_window.py
"""
metrics_window.py — Sliding window untuk akumulasi metrik kedipan.

Window default: 60 detik. Menghitung fitur turunan berikut:
- Blink rate (kedipan/menit), dihaluskan dengan exponential moving average (EMA).
- PERCLOS: persentase waktu mata tertutup dalam window.
- Rata-rata durasi tiap kedipan.
- Variabilitas interval antar-kedipan (coefficient of variation).
- Incomplete blink ratio: proporsi kedipan tidak sempurna.
- Data quality: proporsi frame valid (landmark confidence ≥ 0.5).

Semua fitur ini disediakan untuk composite scoring di modul scoring/
dan sebagai input ke ML inference engine.
"""

import time
from collections import deque
from typing import Deque, Dict, Optional, Tuple

import numpy as np


class MetricsWindow:
    """
    Menyimpan histori kedipan & frame dalam sliding window waktu tertentu,
    menghitung fitur-fitur turunan dengan smoothing (bukan angka instan).
    """

    def __init__(self, window_seconds: int = 60, smoothing_alpha: float = 0.3):
        """
        Args:
            window_seconds: Durasi sliding window dalam detik.
            smoothing_alpha: Koefisien EMA untuk smoothing blink rate
                             (0 < α ≤ 1; makin besar makin responsif).
        """
        self.window_seconds = window_seconds
        self.smoothing_alpha = smoothing_alpha

        # Deque of (timestamp, duration, incomplete_flag)
        self.blink_events: Deque[Tuple[float, float, bool]] = deque()
        # Deque of (timestamp, is_closed)
        self.closed_frame_log: Deque[Tuple[float, bool]] = deque()
        # Deque of (timestamp, is_valid)
        self.valid_frame_log: Deque[Tuple[float, bool]] = deque()

        self._smoothed_rate: Optional[float] = None

    # ─────────────────────────────────────────
    # Data ingestion
    # ─────────────────────────────────────────

    def add_blink(self, event: Dict[str, float]) -> None:
        """Tambahkan event kedipan ke window."""
        incomplete = bool(event.get("incomplete", False))
        self.blink_events.append(
            (event["timestamp"], event["duration"], incomplete)
        )
        self._trim()

    def add_frame(self, timestamp: float, is_closed: bool, is_valid: bool) -> None:
        """Catat status satu frame ke window."""
        self.closed_frame_log.append((timestamp, is_closed))
        self.valid_frame_log.append((timestamp, is_valid))
        self._trim()

    def _trim(self) -> None:
        """Buang entri yang sudah di luar window."""
        cutoff = time.time() - self.window_seconds
        for log in (self.blink_events, self.closed_frame_log, self.valid_frame_log):
            while log and log[0][0] < cutoff:
                log.popleft()

    # ─────────────────────────────────────────
    # Metric calculations
    # ─────────────────────────────────────────

    def data_quality(self) -> float:
        """Proporsi frame valid (landmark terdeteksi) dalam window."""
        if not self.valid_frame_log:
            return 0.0
        valid = sum(1 for _, v in self.valid_frame_log if v)
        return valid / len(self.valid_frame_log)

    def raw_blink_rate_per_minute(self) -> float:
        """Blink rate mentah: jumlah kedipan / elapsed × 60."""
        n = len(self.blink_events)
        elapsed = max(self.window_seconds, 1)
        return (n / elapsed) * 60.0

    def smoothed_blink_rate(self) -> float:
        """
        Blink rate yang dihaluskan dengan Exponential Moving Average.
        Mencegah lonjakan/penurunan drastis karena variabilitas sesaat.
        """
        raw = self.raw_blink_rate_per_minute()
        if self._smoothed_rate is None:
            self._smoothed_rate = raw
        else:
            a = self.smoothing_alpha
            self._smoothed_rate = a * raw + (1 - a) * self._smoothed_rate
        return self._smoothed_rate

    def perclos(self) -> float:
        """
        PERCLOS — Percentage of Eyelid Closure.
        Proporsi frame di mana mata tertutup dalam window.
        Literatur umumnya menggunakan ambang 0.15 (15%) sebagai indikator fatigue tinggi.
        """
        if not self.closed_frame_log:
            return 0.0
        closed = sum(1 for _, c in self.closed_frame_log if c)
        return closed / len(self.closed_frame_log)

    def avg_blink_duration(self) -> float:
        """Rata-rata durasi kedipan (detik) dalam window."""
        if not self.blink_events:
            return 0.0
        durations = [d for _, d, _ in self.blink_events]
        return float(np.mean(durations))

    def interval_variability(self) -> float:
        """
        Coefficient of Variation (CV) dari interval antar-kedipan.
        CV = std / mean. Makin tinggi = pola kedipan makin tidak teratur.
        Butuh minimal 3 kedipan untuk perhitungan yang bermakna.
        """
        timestamps = [t for t, _, _ in self.blink_events]
        if len(timestamps) < 3:
            return 0.0
        intervals = np.diff(timestamps)
        mean_iv = np.mean(intervals)
        if mean_iv == 0:
            return 0.0
        return float(np.std(intervals) / mean_iv)

    def incomplete_blink_ratio(self) -> float:
        """
        Proporsi kedipan tidak sempurna dalam window.
        Secara klinis, incomplete blink lebih relevan terhadap risiko
        mata kering karena tear film tidak terdistribusi sempurna
        ke seluruh permukaan kornea.
        """
        if not self.blink_events:
            return 0.0
        incomplete_count = sum(1 for _, _, inc in self.blink_events if inc)
        return incomplete_count / len(self.blink_events)

    def get_all_metrics(self) -> Dict[str, float]:
        """Kembalikan semua metrik dalam satu dictionary."""
        return {
            "raw_blink_rate": self.raw_blink_rate_per_minute(),
            "smoothed_blink_rate": self.smoothed_blink_rate(),
            "perclos": self.perclos(),
            "avg_blink_duration": self.avg_blink_duration(),
            "interval_variability": self.interval_variability(),
            "incomplete_blink_ratio": self.incomplete_blink_ratio(),
            "data_quality": self.data_quality(),
            "blink_count": len(self.blink_events),
        }
