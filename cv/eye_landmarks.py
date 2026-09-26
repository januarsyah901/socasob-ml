# cv/eye_landmarks.py

import numpy as np
from typing import List, Optional, Tuple

# Index standar 6 titik MediaPipe untuk perhitungan EAR (Eye Aspect Ratio)
# Urutan: [ujung luar/dalam, kelopak atas 1, kelopak atas 2, ujung dalam/luar, kelopak bawah 2, kelopak bawah 1]
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
IRIS_GAZE_EYES = (
    (33, 133, 159, 145, 468),
    (362, 263, 386, 374, 473),
)


def is_looking_at_screen(landmarks: List[Tuple[float, float]]) -> Optional[bool]:
    """Classify central versus averted gaze using normalized iris positions."""
    if len(landmarks) <= 473:
        return None

    for outer_idx, inner_idx, upper_idx, lower_idx, iris_idx in IRIS_GAZE_EYES:
        outer = landmarks[outer_idx]
        inner = landmarks[inner_idx]
        upper = landmarks[upper_idx]
        lower = landmarks[lower_idx]
        iris = landmarks[iris_idx]

        eye_width = abs(inner[0] - outer[0])
        eye_height = abs(lower[1] - upper[1])
        if eye_width <= 0 or eye_height <= 0:
            return None

        horizontal_position = (iris[0] - min(outer[0], inner[0])) / eye_width
        vertical_position = (iris[1] - min(upper[1], lower[1])) / eye_height
        if not (0.25 <= horizontal_position <= 0.75 and 0.15 <= vertical_position <= 0.85):
            return False

    return True

def extract_eye_coordinates(landmarks: List[Tuple[float, float]], 
                            width: int, 
                            height: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mengekstrak koordinat piksel spesifik untuk mata kiri dan kanan dari seluruh landmark wajah.

    Args:
        landmarks (List[Tuple[float, float]]): Daftar 468 koordinat (x, y) ternormalisasi dari MediaPipe.
        width (int): Lebar frame gambar dalam piksel.
        height (int): Tinggi frame gambar dalam piksel.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Dua array NumPy berukuran (6, 2) yang berisi 
        koordinat piksel (x, y) absolut untuk mata kiri dan mata kanan.
    """
    
    # Ekstrak titik untuk mata kiri dan ubah persentase ke nilai piksel (denormalisasi)
    left_eye = np.array([
        (int(landmarks[idx][0] * width), int(landmarks[idx][1] * height))
        for idx in LEFT_EYE_INDICES
    ], dtype=np.int32)

    # Ekstrak titik untuk mata kanan dan ubah persentase ke nilai piksel (denormalisasi)
    right_eye = np.array([
        (int(landmarks[idx][0] * width), int(landmarks[idx][1] * height))
        for idx in RIGHT_EYE_INDICES
    ], dtype=np.int32)

    return left_eye, right_eye