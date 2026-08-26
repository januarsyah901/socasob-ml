# cv/visualization.py

import cv2
import numpy as np
from typing import Dict, Any, Optional

class Visualizer:
    """
    Modul untuk menggambar anotasi visual (overlay) pada frame.
    
    Menampilkan metrik komposit: EAR, Status Kelopak, Blink Count, Blink Rate,
    PERCLOS, Skor Kelelahan Komposit, dan Status Sistem.
    """

    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.55
        self.thickness = 2
        
        # Warna dalam format BGR (Blue, Green, Red)
        self.color_text = (240, 240, 240)       # Putih keabuan
        self.color_safe = (60, 220, 60)         # Hijau (Aman)
        self.color_warning_light = (0, 180, 255)# Kuning-Oranye (Peringatan Ringan)
        self.color_warning_dark = (50, 50, 255) # Merah (Peringatan Berat)
        self.color_eye = (0, 255, 255)          # Kuning (Outline Mata)

    def draw_annotations(self, 
                          frame: np.ndarray, 
                          features: Dict[str, Any],
                          left_eye: Optional[np.ndarray] = None,
                          right_eye: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Menggambar overlay teks (metrik API) dan garis poligon landmark pada frame.
        """
        annotated_frame = frame.copy()

        # 1. Gambar outline mata jika wajah terdeteksi
        if features.get("face_detected") and left_eye is not None and right_eye is not None:
            cv2.polylines(annotated_frame, [left_eye], isClosed=True, color=self.color_eye, thickness=1)
            cv2.polylines(annotated_frame, [right_eye], isClosed=True, color=self.color_eye, thickness=1)

        # 2. Ekstrak data metrik
        ear = features.get("ear", 0.0)
        status = features.get("eye_status", "Unknown")
        blinks = features.get("blink_count", 0)
        rate = features.get("blink_rate", 0.0)
        perclos = features.get("perclos", 0.0)
        score = features.get("composite_score", 0.0)
        system_status = features.get("system_status") or features.get("health_status", "Aman")
        fps = features.get("fps", 0.0)

        # Tentukan warna status
        if "Berat" in system_status:
            status_color = self.color_warning_dark
        elif "Ringan" in system_status or "Risiko" in system_status:
            status_color = self.color_warning_light
        elif "Tidak Valid" in system_status:
            status_color = (150, 150, 150)
        else:
            status_color = self.color_safe

        # 3. Baris teks overlay
        texts = [
            (f"FPS: {fps:.1f} | Face: {'Yes' if features.get('face_detected') else 'No'}", self.color_text),
            (f"EAR: {ear:.2f} ({status})", self.color_warning_dark if status == "Closed" else self.color_text),
            (f"Blinks: {blinks} | Rate: {rate:.1f}/min", self.color_text),
            (f"PERCLOS: {perclos*100:.1f}%", self.color_text),
            (f"Status: {system_status}", status_color)
        ]

        # 4. Gambar teks dengan background transparan untuk keterbacaan tinggi
        y_offset = 25
        line_height = 24
        
        # Semi-transparent background banner
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (10, 8), (380, 10 + len(texts) * line_height), (20, 20, 30), -1)
        cv2.addWeighted(overlay, 0.65, annotated_frame, 0.35, 0, annotated_frame)

        for i, (text, color) in enumerate(texts):
            cv2.putText(
                annotated_frame,
                text,
                (18, y_offset + (i * line_height)),
                self.font,
                self.font_scale,
                color,
                1,
                cv2.LINE_AA
            )

        return annotated_frame