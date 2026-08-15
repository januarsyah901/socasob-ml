# cv/visualization.py

import cv2
import numpy as np
from typing import Dict, Any, Optional

class Visualizer:
    """
    Modul untuk menggambar anotasi visual (overlay) pada frame.
    
    Sesuai prinsip Single Responsibility, kelas ini TIDAK melakukan
    perhitungan matematis apa pun. Tugasnya murni hanya mengambil
    data metrik yang sudah ada dan menampilkannya di atas frame gambar
    sebagai output visual untuk endpoint /video_feed.
    """

    def __init__(self):
        """
        Inisialisasi gaya visual (font, ketebalan, dan warna).
        """
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.thickness = 2
        
        # Warna dalam format BGR (Blue, Green, Red)
        self.color_text = (0, 255, 0)       # Hijau
        self.color_warning = (0, 0, 255)    # Merah
        self.color_eye = (0, 255, 255)      # Kuning

    def draw_annotations(self, 
                         frame: np.ndarray, 
                         features: Dict[str, Any],
                         left_eye: Optional[np.ndarray] = None,
                         right_eye: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Menggambar overlay teks (metrik API) dan garis poligon landmark pada frame.

        Args:
            frame (np.ndarray): Matriks gambar BGR asli dari kamera.
            features (Dict[str, Any]): Dictionary berisi metrik (EAR, status, dll).
            left_eye (Optional[np.ndarray]): Matriks koordinat piksel mata kiri.
            right_eye (Optional[np.ndarray]): Matriks koordinat piksel mata kanan.

        Returns:
            np.ndarray: Frame baru yang sudah ditambahkan anotasi visual.
        """
        # Copy frame untuk mencegah modifikasi pada frame asli yang mungkin diakses thread lain
        annotated_frame = frame.copy()

        # 1. Gambar outline/garis mata jika wajah terdeteksi dan koordinat tersedia
        if features.get("face_detected") and left_eye is not None and right_eye is not None:
            cv2.polylines(annotated_frame, [left_eye], isClosed=True, color=self.color_eye, thickness=1)
            cv2.polylines(annotated_frame, [right_eye], isClosed=True, color=self.color_eye, thickness=1)

        # 2. Ekstrak data dari dictionary
        ear = features.get("ear", 0.0)
        status = features.get("eye_status", "Unknown")
        blinks = features.get("blink_count", 0)
        fps = features.get("fps", 0.0)

        # Tentukan warna peringatan jika mata sedang tertutup
        status_color = self.color_warning if status == "Closed" else self.color_text

        # 3. Siapkan baris teks untuk diletakkan di sudut kiri atas
        texts = [
            f"FPS: {fps}",
            f"EAR: {ear:.2f}",
            f"Status: {status}",
            f"Blinks: {blinks}",
            f"Face: {'Yes' if features.get('face_detected') else 'No'}"
        ]

        # 4. Gambar teks secara berurutan
        y_offset = 30
        for i, text in enumerate(texts):
            color = status_color if "Status" in text else self.color_text
            cv2.putText(
                annotated_frame,
                text,
                (20, y_offset + (i * 30)),
                self.font,
                self.font_scale,
                color,
                self.thickness
            )

        return annotated_frame