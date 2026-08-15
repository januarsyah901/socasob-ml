# cv/eye_landmarks.py

import numpy as np
from typing import List, Tuple

# Index standar 6 titik MediaPipe untuk perhitungan EAR (Eye Aspect Ratio)
# Urutan: [ujung luar/dalam, kelopak atas 1, kelopak atas 2, ujung dalam/luar, kelopak bawah 2, kelopak bawah 1]
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

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