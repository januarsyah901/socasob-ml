# scoring/active_myopia_guard.py
"""
active_myopia_guard.py — Modul B1: Intervensi Aktif Real-Time.

Dua cabang paralel yang berjalan kontinu per detik:

1. Cabang Jarak:
   - Estimasi jarak mata-layar via pinhole camera model (di vision/distance_estimator.py).
   - Ambang: ≥ 50 cm (≈20 inci) = aman; < 50 cm → peringatan instan "Jarak terlalu dekat".
   - Peringatan INSTAN tanpa hysteresis — karena risiko ergonomi memerlukan respons cepat.

2. Cabang Durasi (Aturan 20-20-20):
   - Timer kontinu selama wajah terdeteksi di depan layar.
   - Begitu mencapai 20 menit → trigger peringatan istirahat 20 detik.
   - Arahkan melihat objek sejauh 20 kaki (6 meter).
   - Setelah 20 detik → reminder lanjutkan aktivitas, reset timer ke nol.

Referensi:
    Ambang berdasarkan rekomendasi ergonomi & aturan 20-20-20 dari
    Kaur dkk. (2022, Ophthalmology and Therapy).
"""

import time
from enum import Enum
from typing import Dict, Optional


class BreakState(Enum):
    """Status cabang durasi (aturan 20-20-20)."""
    ACTIVE = "active"           # Pengguna sedang bekerja, timer berjalan
    BREAK_NEEDED = "break_needed"   # 20 menit tercapai, peringatan aktif
    ON_BREAK = "on_break"        # Sedang istirahat 20 detik


class ActiveMyopiaGuard:
    """
    Modul B1: Sistem peringatan aktif real-time untuk jarak & durasi layar.

    Didesain untuk dipanggil setiap frame/detik dari pipeline utama.
    """

    # Durasi kerja sebelum istirahat (detik) — aturan 20-20-20
    WORK_DURATION_SEC: float = 20 * 60  # 20 menit = 1200 detik

    # Durasi istirahat yang diperlukan (detik)
    BREAK_DURATION_SEC: float = 20.0  # 20 detik

    # Ambang jarak aman (cm) — Kaur dkk. (2022, Ophthalmology and Therapy)
    SAFE_DISTANCE_CM: float = 50.0

    def __init__(self):
        # Cabang Durasi
        self._work_start_time: Optional[float] = None
        self._break_start_time: Optional[float] = None
        self._break_state = BreakState.ACTIVE
        self._total_work_seconds: float = 0.0  # akumulasi sesi

        # Cabang Jarak — counter peringatan
        self._distance_warning_count: int = 0
        self._break_reminder_count: int = 0

        # Timestamp terakhir face detected (untuk deteksi absence)
        self._last_face_time: Optional[float] = None

    def update(
        self,
        face_detected: bool,
        distance_cm: Optional[float],
        timestamp: Optional[float] = None,
    ) -> Dict[str, object]:
        """
        Proses satu frame/tick dan kembalikan status kedua cabang.

        Args:
            face_detected: True jika wajah terdeteksi di frame ini.
            distance_cm: Estimasi jarak mata-layar (cm), None jika tidak tersedia.
            timestamp: Waktu saat ini (detik, monotonic). Default: time.time().

        Returns:
            Dict dengan field:
                distance_cm: float | None
                distance_warning: bool — True jika jarak < 50 cm
                break_state: str — "active" / "break_needed" / "on_break"
                work_elapsed_sec: float — detik sejak mulai bekerja
                break_remaining_sec: float — sisa waktu istirahat (0 jika tidak break)
                break_reminder_count: int — total reminder istirahat yang dipicu
                distance_warning_count: int — total peringatan jarak
        """
        now = timestamp if timestamp else time.time()

        # ─────────────────────────────────────
        # Cabang Jarak (instan, tanpa hysteresis)
        # ─────────────────────────────────────
        distance_warning = False
        if distance_cm is not None and distance_cm < self.SAFE_DISTANCE_CM:
            distance_warning = True
            self._distance_warning_count += 1

        # ─────────────────────────────────────
        # Cabang Durasi (aturan 20-20-20)
        # ─────────────────────────────────────
        work_elapsed = 0.0
        break_remaining = 0.0

        if face_detected:
            self._last_face_time = now

            if self._break_state == BreakState.ACTIVE:
                # Mulai timer jika belum
                if self._work_start_time is None:
                    self._work_start_time = now

                work_elapsed = now - self._work_start_time

                # Cek apakah 20 menit tercapai
                if work_elapsed >= self.WORK_DURATION_SEC:
                    self._break_state = BreakState.BREAK_NEEDED
                    self._break_reminder_count += 1

            elif self._break_state == BreakState.BREAK_NEEDED:
                # Tunggu pengguna mulai istirahat (= wajah menghilang)
                # Sementara wajah masih ada, peringatan tetap aktif
                work_elapsed = now - (self._work_start_time or now)

            elif self._break_state == BreakState.ON_BREAK:
                # Pengguna kembali sebelum 20 detik selesai?
                # Hitung sisa waktu break
                if self._break_start_time:
                    elapsed_break = now - self._break_start_time
                    if elapsed_break >= self.BREAK_DURATION_SEC:
                        # Break selesai → reset
                        self._reset_work_timer(now)
                    else:
                        break_remaining = self.BREAK_DURATION_SEC - elapsed_break

        else:
            # Wajah tidak terdeteksi
            if self._break_state == BreakState.BREAK_NEEDED:
                # Pengguna mulai istirahat (menjauh dari layar)
                self._break_state = BreakState.ON_BREAK
                self._break_start_time = now

            elif self._break_state == BreakState.ON_BREAK:
                if self._break_start_time:
                    elapsed_break = now - self._break_start_time
                    if elapsed_break >= self.BREAK_DURATION_SEC:
                        # Break selesai → reset
                        self._reset_work_timer(now)
                    else:
                        break_remaining = self.BREAK_DURATION_SEC - elapsed_break

            # Jika ACTIVE dan wajah hilang, pause timer (tidak reset)

        return {
            "distance_cm": distance_cm,
            "distance_warning": distance_warning,
            "break_state": self._break_state.value,
            "work_elapsed_sec": round(work_elapsed, 1),
            "break_remaining_sec": round(break_remaining, 1),
            "break_reminder_count": self._break_reminder_count,
            "distance_warning_count": self._distance_warning_count,
        }

    def _reset_work_timer(self, now: float) -> None:
        """Reset timer kerja setelah istirahat selesai."""
        self._break_state = BreakState.ACTIVE
        self._work_start_time = now
        self._break_start_time = None

    def get_total_work_seconds(self) -> float:
        """Total detik pengguna bekerja di depan layar dalam sesi ini."""
        if self._work_start_time is None:
            return 0.0
        return time.time() - self._work_start_time

    @property
    def distance_warning_count(self) -> int:
        return self._distance_warning_count

    @property
    def break_reminder_count(self) -> int:
        return self._break_reminder_count
