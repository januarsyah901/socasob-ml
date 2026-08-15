# cv/ear.py

import numpy as np

def calculate_ear(eye_points: np.ndarray) -> float:
    """
    Menghitung Eye Aspect Ratio (EAR) berdasarkan 6 titik landmark mata.
    
    Fungsi ini bersifat stateless dan murni melakukan perhitungan matematis
    menggunakan jarak Euclidean antar titik.

    Args:
        eye_points (np.ndarray): Matriks NumPy berukuran (6, 2) yang berisi
                                 koordinat piksel (x, y) dari satu mata.
                                 Urutan titik: [p1, p2, p3, p4, p5, p6].

    Returns:
        float: Rasio EAR. Akan mengembalikan 0.0 jika terjadi pembagian dengan nol.
    """
    # Menghitung jarak Euclidean vertikal (antara kelopak atas dan bawah)
    # Titik p2 - p6 (indeks 1 dan 5)
    v1 = np.linalg.norm(eye_points[1] - eye_points[5])
    # Titik p3 - p5 (indeks 2 dan 4)
    v2 = np.linalg.norm(eye_points[2] - eye_points[4])

    # Menghitung jarak Euclidean horizontal (antara ujung dalam dan luar mata)
    # Titik p1 - p4 (indeks 0 dan 3)
    h = np.linalg.norm(eye_points[0] - eye_points[3])

    # Mencegah error pembagian dengan nol (division by zero)
    if h == 0.0:
        return 0.0

    # Menghitung Eye Aspect Ratio
    ear = (v1 + v2) / (2.0 * h)
    
    return float(ear)