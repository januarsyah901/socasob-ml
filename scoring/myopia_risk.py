# scoring/myopia_risk.py
"""
myopia_risk.py — Modul B2: Estimasi Risiko Miopia Kumulatif.

Pipeline:
    Session Duration Tracker (akumulasi screen time harian)
    → Lookup/interpolasi kurva dosis-respons non-linear
    → Estimasi persentase peningkatan risiko

Tabel lookup (interpolasi linear antar titik, ekstrapolasi melandai
di luar 5 jam — kurva aslinya sigmoid, bukan linear terus):

    Jam/hari    OR      Peningkatan risiko
    0.5         1.01    +1%
    1.0         1.05    +5%
    1.5         1.14    +14%
    2.0         1.29    +29%
    2.5         1.47    +47%
    3.0         1.65    +65%
    3.5         1.82    +82%
    4.0         1.97    +97%
    4.5         2.11    +111%
    5.0         2.24    +124%

Sumber: Ha dkk. (2025), JAMA Network Open, 45 studi, n=335.524.

DISCLAIMER: Nilai ini bersifat asosiatif dari data observasional
(GRADE-low certainty, I²=99%), bukan prediksi kausal individual.
Peningkatan risiko dihitung sebagai (OR - 1) × 100%.
"""

import time
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────
# Tabel Dosis-Respons (Ha dkk., 2025)
# (jam_per_hari, odds_ratio)
# ─────────────────────────────────────────────
DOSE_RESPONSE_TABLE: List[Tuple[float, float]] = [
    (0.0, 1.00),   # baseline: tidak ada peningkatan risiko
    (0.5, 1.01),
    (1.0, 1.05),
    (1.5, 1.14),
    (2.0, 1.29),
    (2.5, 1.47),
    (3.0, 1.65),
    (3.5, 1.82),
    (4.0, 1.97),
    (4.5, 2.11),
    (5.0, 2.24),
]


def _interpolate_risk(hours: float) -> float:
    """
    Interpolasi/ekstrapolasi risiko dari tabel dosis-respons.

    - Di dalam range (0–5 jam): interpolasi linear antar titik.
    - Di luar 5 jam: ekstrapolasi melandai (logaritmik), bukan linear terus,
      karena kurva aslinya berbentuk sigmoid yang mulai mendatar.

    Args:
        hours: Screen time harian dalam jam.

    Returns:
        Odds Ratio (OR). Peningkatan risiko = (OR - 1) × 100%.
    """
    if hours <= 0:
        return 1.0

    table = DOSE_RESPONSE_TABLE

    # Cari interval yang sesuai untuk interpolasi linear
    for i in range(len(table) - 1):
        h0, or0 = table[i]
        h1, or1 = table[i + 1]
        if h0 <= hours <= h1:
            # Interpolasi linear antar dua titik terdekat
            t = (hours - h0) / (h1 - h0)
            return or0 + t * (or1 - or0)

    # Di luar 5 jam → ekstrapolasi melandai (logaritmik)
    # Gunakan model: OR = OR_5h + k × ln(hours / 5)
    # di mana k dihitung dari slope terakhir yang didampingi decay
    last_h, last_or = table[-1]
    prev_h, prev_or = table[-2]

    # Slope terakhir (per jam)
    last_slope = (last_or - prev_or) / (last_h - prev_h)

    # Ekstrapolasi logaritmik: slope melandai secara alami
    extra_hours = hours - last_h
    if extra_hours <= 0:
        return last_or

    # OR = last_or + last_slope × ln(1 + extra_hours)
    # ln(1+x) tumbuh sangat lambat untuk x besar → efek mendatar
    return last_or + last_slope * np.log(1 + extra_hours)


def estimate_risk_percentage(hours: float) -> float:
    """
    Hitung persentase peningkatan risiko miopia dari screen time harian.

    Args:
        hours: Total screen time hari ini dalam jam.

    Returns:
        Peningkatan risiko dalam persen (misal 97.0 = +97%).
    """
    odds_ratio = _interpolate_risk(hours)
    return round((odds_ratio - 1) * 100, 1)


class MyopiaRiskEstimator:
    """
    Modul B2: Tracker screen time harian dan estimator risiko miopia kumulatif.

    DISCLAIMER: Nilai risiko bersifat asosiatif dari data observasional
    (GRADE-low certainty, I²=99%), bukan prediksi kausal individual.
    Sumber: Ha dkk. (2025), JAMA Network Open, 45 studi, n=335.524.
    """

    def __init__(self):
        self._session_start: Optional[float] = None
        self._accumulated_seconds: float = 0.0
        self._last_active_time: Optional[float] = None
        # Gap > 5 menit dianggap break (tidak dihitung screen time)
        self._inactivity_threshold: float = 300.0

    def start_session(self) -> None:
        """Mulai atau lanjutkan sesi screen time."""
        now = time.time()
        if self._session_start is None:
            self._session_start = now
        self._last_active_time = now

    def tick(self, face_detected: bool) -> None:
        """
        Dipanggil setiap frame/detik.
        Akumulasi screen time hanya jika wajah terdeteksi.
        """
        now = time.time()

        if face_detected:
            if self._last_active_time is not None:
                delta = now - self._last_active_time
                # Hanya tambahkan jika gap < inactivity threshold
                if delta < self._inactivity_threshold:
                    self._accumulated_seconds += delta
            self._last_active_time = now
        else:
            # Wajah tidak terdeteksi — jangan akumulasi
            pass

    def get_screen_time_hours(self) -> float:
        """Total screen time hari ini dalam jam."""
        return self._accumulated_seconds / 3600.0

    def get_screen_time_minutes(self) -> float:
        """Total screen time hari ini dalam menit."""
        return self._accumulated_seconds / 60.0

    def get_risk(self) -> Dict[str, float]:
        """
        Hitung estimasi risiko miopia dari screen time terakumulasi.

        Returns:
            Dict:
                screen_time_hours: float
                screen_time_minutes: float
                risk_percentage: float — peningkatan risiko dalam %
                odds_ratio: float
        """
        hours = self.get_screen_time_hours()
        risk_pct = estimate_risk_percentage(hours)
        odds_ratio = _interpolate_risk(hours)

        return {
            "screen_time_hours": round(hours, 2),
            "screen_time_minutes": round(hours * 60, 1),
            "risk_percentage": risk_pct,
            "odds_ratio": round(odds_ratio, 3),
        }

    def reset_daily(self) -> None:
        """Reset akumulasi screen time (dipanggil saat pergantian hari)."""
        self._accumulated_seconds = 0.0
        self._session_start = time.time()
        self._last_active_time = None
