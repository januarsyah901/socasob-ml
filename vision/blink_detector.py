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
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

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


class EARSmoother:
    """Smoothing EAR EMA dengan histori pendek untuk menekan noise landmark."""

    def __init__(self, window_size: int = 3, alpha: float = 0.5):
        self.values: Deque[float] = deque(maxlen=window_size)
        self.alpha = alpha
        self._value: Optional[float] = None

    def update(self, ear_value: float) -> float:
        self.values.append(float(ear_value))
        if self._value is None:
            self._value = float(ear_value)
        else:
            self._value = self.alpha * float(ear_value) + (1.0 - self.alpha) * self._value
        return self._value

    def reset(self) -> None:
        self.values.clear()
        self._value = None


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
    EAR_CLOSE_THRESHOLD: float = 0.21
    EAR_OPEN_THRESHOLD: float = 0.26
    INCOMPLETE_EAR_THRESHOLD: float = 0.15
    MIN_CLOSED_FRAMES: int = 3
    BLINK_COOLDOWN_FRAMES: int = 5

    def __init__(
        self,
        ear_threshold: float = 0.21,
        min_closed_frames: int = 3,
        ear_open_threshold: float = 0.26,
        smoothing_window: int = 3,
        cooldown_frames: int = 5,
    ):
        """
        Args:
            ear_threshold: Batas EAR penutupan (dipertahankan untuk kompatibilitas).
            min_closed_frames: Jumlah frame berturut-turut di bawah threshold
                               sebelum state beralih ke CLOSED.
        """
        self.ear_threshold = ear_threshold
        self.close_threshold = ear_threshold
        self.open_threshold = ear_open_threshold
        self.min_closed_frames = max(min_closed_frames, self.MIN_CLOSED_FRAMES)
        self.cooldown_frames = cooldown_frames
        self._cooldown_remaining = 0
        self.smoother = EARSmoother(window_size=smoothing_window)

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
        # Data quality gate: frame buruk tidak mengubah state atau statistik.
        if face_confidence < self.MIN_CONFIDENCE:
            return None

        smoothed_ear = self.smoother.update(ear_value)
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            self.state = EyeState.OPEN
            self._closed_frame_count = 0
            return None

        if self.state == EyeState.OPEN and smoothed_ear < self.close_threshold:
            self.state = EyeState.CLOSING
            self._blink_start_time = timestamp
            self._closed_frame_count = 1
            self._min_ear_during_blink = smoothed_ear
        elif self.state == EyeState.CLOSING:
            if smoothed_ear > self.open_threshold:
                self.state = EyeState.OPEN
                self._closed_frame_count = 0
                self._blink_start_time = None
                self._min_ear_during_blink = 1.0
            else:
                self._closed_frame_count += 1
                self._min_ear_during_blink = min(self._min_ear_during_blink, smoothed_ear)
                if self._closed_frame_count >= self.min_closed_frames:
                    self.state = EyeState.CLOSED
        elif self.state == EyeState.CLOSED:
            # Track EAR minimum selama kedipan (untuk incomplete blink detection)
            if smoothed_ear <= self.close_threshold:
                self._min_ear_during_blink = min(self._min_ear_during_blink, smoothed_ear)
            elif smoothed_ear > self.open_threshold:
                start = self._blink_start_time if self._blink_start_time else timestamp
                duration = timestamp - start
                incomplete = self._min_ear_during_blink > self.INCOMPLETE_EAR_THRESHOLD
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
                self._cooldown_remaining = self.cooldown_frames
                return event

        return None
