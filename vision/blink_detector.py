# vision/blink_detector.py
"""
blink_detector.py — Deteksi kedipan mata dari nilai EAR (Eye Aspect Ratio).

Pipeline:
  MediaPipe landmarks → extract_eye_coordinates() → calculate_ear()
  → BlinkEventDetector (state machine OPEN→CLOSING→CLOSED→OPEN)

Fitur utama:
- Data quality gate: frame dengan confidence < 0.5 diabaikan, bukan
  dianggap "mata tidak berkedip" (mencegah false-positive fatigue).
- Incomplete blink flag: kedipan yang EAR minimum-nya tidak mendekati 0
  (min_ear > threshold × 0.5) ditandai incomplete — secara klinis lebih
  relevan terhadap risiko mata kering daripada frekuensi semata.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

# ─────────────────────────────────────────────
# MediaPipe Face Mesh: indeks landmark 6-titik EAR
# Urutan: [outer corner, upper1, upper2, inner corner, lower2, lower1]
# ─────────────────────────────────────────────
LEFT_EYE_INDICES: List[int] = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES: List[int] = [362, 385, 387, 263, 373, 380]

# Indeks ujung luar mata kiri & kanan — dipakai oleh DistanceEstimator
LEFT_EYE_OUTER: int = 33
RIGHT_EYE_OUTER: int = 263


# ─────────────────────────────────────────────
# Eye Aspect Ratio (EAR)
# ─────────────────────────────────────────────
def calculate_ear(eye_points: np.ndarray) -> float:
    """
    Menghitung Eye Aspect Ratio (EAR) dari 6 titik landmark mata.

    EAR = (|p2-p6| + |p3-p5|) / (2 × |p1-p4|)

    Args:
        eye_points: Array (6, 2) koordinat piksel (x, y).

    Returns:
        float: Nilai EAR. 0.0 jika terjadi division by zero.
    """
    v1 = np.linalg.norm(eye_points[1] - eye_points[5])
    v2 = np.linalg.norm(eye_points[2] - eye_points[4])
    h = np.linalg.norm(eye_points[0] - eye_points[3])

    if h == 0.0:
        return 0.0

    return float((v1 + v2) / (2.0 * h))


def extract_eye_coordinates(
    landmarks: List[Tuple[float, float]],
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mengekstrak koordinat piksel absolut untuk mata kiri dan kanan
    dari 468 landmark ternormalisasi MediaPipe Face Mesh.

    Args:
        landmarks: List of (x, y) normalized [0..1] dari MediaPipe.
        width: Lebar frame (piksel).
        height: Tinggi frame (piksel).

    Returns:
        Tuple (left_eye, right_eye) masing-masing array (6, 2) int32.
    """
    left_eye = np.array(
        [(int(landmarks[i][0] * width), int(landmarks[i][1] * height))
         for i in LEFT_EYE_INDICES],
        dtype=np.int32,
    )
    right_eye = np.array(
        [(int(landmarks[i][0] * width), int(landmarks[i][1] * height))
         for i in RIGHT_EYE_INDICES],
        dtype=np.int32,
    )
    return left_eye, right_eye


# ─────────────────────────────────────────────
# Eye State (untuk state machine kedipan)
# ─────────────────────────────────────────────
class EyeState(Enum):
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"


# ─────────────────────────────────────────────
# Blink Event Detector (State Machine)
# ─────────────────────────────────────────────
class BlinkEventDetector:
    """
    Mengubah aliran nilai EAR per-frame menjadi event kedipan diskrit
    lengkap dengan durasi dan incomplete-blink flag.

    State machine: OPEN → CLOSING → CLOSED → OPEN (emit event)

    Frame dengan confidence landmark rendah (< 0.5) diabaikan
    sesuai prinsip data quality gating — bukan dianggap "tidak berkedip".
    """

    # Confidence minimum untuk memproses frame (Bagian 3: Data Quality Gating)
    MIN_CONFIDENCE: float = 0.5

    def __init__(
        self,
        ear_threshold: float = 0.21,
        min_closed_frames: int = 2,
    ):
        """
        Args:
            ear_threshold: Batas EAR di bawah mana mata dianggap tertutup.
            min_closed_frames: Jumlah frame berturut-turut di bawah threshold
                               sebelum state beralih ke CLOSED.
        """
        self.ear_threshold = ear_threshold
        self.min_closed_frames = min_closed_frames

        self.state = EyeState.OPEN
        self._closed_frame_count: int = 0
        self._blink_start_time: Optional[float] = None
        self._min_ear_during_blink: float = 1.0  # track EAR minimum per kedipan

    def update(
        self,
        ear_value: float,
        face_confidence: float,
        timestamp: float,
    ) -> Optional[Dict[str, float]]:
        """
        Proses satu frame EAR dan kembalikan event kedipan jika terjadi.

        Args:
            ear_value: Nilai EAR rata-rata kedua mata.
            face_confidence: Skor confidence deteksi wajah [0..1].
            timestamp: Waktu frame (detik, monotonic).

        Returns:
            Dict blink event jika kedipan selesai:
                {"duration": float, "timestamp": float, "incomplete": bool}
            None jika tidak ada event.
        """
        # Data quality gate (Bagian 3)
        if face_confidence < self.MIN_CONFIDENCE:
            return None

        if ear_value < self.ear_threshold:
            # Track EAR minimum selama kedipan (untuk incomplete blink detection)
            self._min_ear_during_blink = min(self._min_ear_during_blink, ear_value)

            if self.state == EyeState.OPEN:
                self.state = EyeState.CLOSING
                self._blink_start_time = timestamp
                self._closed_frame_count = 1
                self._min_ear_during_blink = ear_value
            elif self.state == EyeState.CLOSING:
                self._closed_frame_count += 1
                if self._closed_frame_count >= self.min_closed_frames:
                    self.state = EyeState.CLOSED
            # Jika sudah CLOSED, tetap di CLOSED selama EAR di bawah threshold

        else:
            if self.state in (EyeState.CLOSING, EyeState.CLOSED):
                # Transisi kembali ke OPEN → emit blink event
                start = self._blink_start_time if self._blink_start_time else timestamp
                duration = timestamp - start

                # Incomplete blink flag:
                # Jika EAR minimum selama kedipan masih > 50% threshold,
                # berarti mata tidak benar-benar tertutup sempurna.
                # Secara klinis, ini lebih relevan terhadap risiko mata kering
                # (tear film tidak terdistribusi sempurna).
                incomplete = self._min_ear_during_blink > (self.ear_threshold * 0.5)

                event = {
                    "duration": duration,
                    "timestamp": timestamp,
                    "incomplete": incomplete,
                    "min_ear": self._min_ear_during_blink,
                }

                self.state = EyeState.OPEN
                self._closed_frame_count = 0
                self._blink_start_time = None
                self._min_ear_during_blink = 1.0
                return event

            self.state = EyeState.OPEN

        return None
