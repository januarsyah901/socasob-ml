# vision/distance_estimator.py
"""
distance_estimator.py — Estimasi jarak mata-layar via model kamera pinhole.

Rumus:
    jarak_cm = (jarak_interokular_asumsi_cm × focal_length_px)
               / jarak_interokular_terukur_px

Menggunakan landmark mata yang sama dari MediaPipe Face Mesh
(outer corners: index 33 untuk mata kiri, 263 untuk mata kanan).

Asumsi:
- Jarak interokular rata-rata dewasa: 6.3 cm (Dodgson, 2004)
- Focal length default diestimasi dari lebar frame (≈ frame_width untuk
  field of view ~60°, tipikal webcam consumer-grade)

Referensi ambang:
- ≥ 50 cm (≈20 inci) = aman
- < 50 cm → trigger peringatan "Jarak terlalu dekat"
  Berdasarkan rekomendasi ergonomi Kaur dkk. (2022, Ophthalmology and Therapy).
"""

from typing import List, Optional, Tuple

import numpy as np

from vision.blink_detector import LEFT_EYE_OUTER, RIGHT_EYE_OUTER


class DistanceEstimator:
    """
    Estimasi jarak mata-layar menggunakan model kamera pinhole
    berbasis jarak interokular yang terukur dari landmark.
    """

    # Jarak interokular rata-rata dewasa (cm) — Dodgson (2004)
    ASSUMED_INTEROCULAR_CM: float = 6.3

    # Ambang jarak aman (cm) — Kaur dkk. (2022, Ophthalmology and Therapy)
    SAFE_DISTANCE_CM: float = 50.0

    def __init__(
        self,
        frame_width: int = 640,
        focal_length_px: Optional[float] = None,
    ):
        """
        Args:
            frame_width: Lebar frame kamera dalam piksel.
            focal_length_px: Focal length kamera dalam piksel.
                             Jika None, diestimasi dari frame_width
                             (≈ frame_width untuk FoV ~60°, tipikal webcam).
        """
        self._frame_width = frame_width
        # Estimasi focal length: untuk FoV horizontal ~60°,
        # f ≈ (w/2) / tan(30°) ≈ w × 0.866. Kita gunakan ~w sebagai aproksimasi.
        self._focal_length = focal_length_px if focal_length_px else float(frame_width)

    def update_frame_width(self, frame_width: int) -> None:
        """Update focal length jika resolusi kamera berubah."""
        if frame_width != self._frame_width and frame_width > 0:
            self._frame_width = frame_width
            self._focal_length = float(frame_width)

    def estimate(
        self,
        landmarks: List[Tuple[float, float]],
        frame_width: int,
        frame_height: int,
    ) -> Optional[float]:
        """
        Menghitung estimasi jarak mata-layar dari landmark wajah.

        Args:
            landmarks: 468 koordinat (x, y) ternormalisasi dari MediaPipe.
            frame_width: Lebar frame dalam piksel.
            frame_height: Tinggi frame dalam piksel.

        Returns:
            Jarak dalam cm, atau None jika gagal menghitung.
        """
        self.update_frame_width(frame_width)

        # Ambil posisi piksel ujung luar mata kiri dan kanan
        lx = landmarks[LEFT_EYE_OUTER][0] * frame_width
        ly = landmarks[LEFT_EYE_OUTER][1] * frame_height
        rx = landmarks[RIGHT_EYE_OUTER][0] * frame_width
        ry = landmarks[RIGHT_EYE_OUTER][1] * frame_height

        # Jarak interokular terukur (piksel)
        measured_px = np.sqrt((rx - lx) ** 2 + (ry - ly) ** 2)

        if measured_px < 1.0:
            return None  # landmark terlalu dekat / noise

        # Model pinhole
        distance_cm = (self.ASSUMED_INTEROCULAR_CM * self._focal_length) / measured_px

        return round(float(distance_cm), 1)

    def is_too_close(self, distance_cm: Optional[float]) -> bool:
        """
        Mengecek apakah jarak di bawah ambang aman.

        Peringatan ini bersifat instan (tanpa hysteresis) sesuai spesifikasi
        Modul B1 — karena jarak terlalu dekat perlu respons cepat.
        """
        if distance_cm is None:
            return False
        return distance_cm < self.SAFE_DISTANCE_CM
