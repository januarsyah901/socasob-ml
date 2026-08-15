# cv/face_mesh.py

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Optional
from utils.logger import get_logger

logger = get_logger(__name__)

class FaceMeshDetector:
    """
    Wrapper untuk MediaPipe Face Mesh.
    
    Kelas ini bertanggung jawab murni untuk menginisialisasi model MediaPipe
    dan melakukan inference pada frame BGR untuk mendeteksi 468 titik landmark wajah.
    Sesuai prinsip Dependency Inversion, kelas ini tidak tahu-menahu soal
    sumber kamera atau bagaimana visualisasinya nanti.
    """

    def __init__(self, 
                 max_num_faces: int = 1, 
                 min_detection_confidence: float = 0.5, 
                 min_tracking_confidence: float = 0.5):
        """
        Inisialisasi modul MediaPipe Face Mesh.

        Args:
            max_num_faces (int): Jumlah maksimal wajah yang dideteksi (default: 1).
            min_detection_confidence (float): Threshold minimum untuk deteksi awal.
            min_tracking_confidence (float): Threshold minimum untuk tracking frame selanjutnya.
        """
        logger.info("Menginisialisasi MediaPipe Face Mesh...")
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=False,  # Set False untuk membatasi ke 468 titik standar
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def process(self, frame: np.ndarray) -> Optional[List[Tuple[float, float]]]:
        """
        Melakukan inference Face Mesh pada frame gambar.

        Args:
            frame (np.ndarray): Matriks gambar BGR dari kamera.

        Returns:
            Optional[List[Tuple[float, float]]]: Daftar 468 koordinat (x, y) dalam format 
            ternormalisasi [0.0, 1.0]. Mengembalikan None jika wajah tidak terdeteksi.
        """
        if frame is None:
            return None

        # MediaPipe memproses gambar dalam format RGB, sedangkan OpenCV menggunakan BGR
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Lakukan deteksi landmark
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None
            
        # Ambil landmark dari wajah pertama yang terdeteksi
        face_landmarks = results.multi_face_landmarks[0]
        
        # Ekstrak koordinat x dan y ternormalisasi (mengabaikan z / kedalaman)
        landmarks = [(lm.x, lm.y) for lm in face_landmarks.landmark]
        
        return landmarks
        
    def release(self) -> None:
        """
        Menutup model dan membebaskan memori yang digunakan oleh MediaPipe.
        """
        logger.info("Melepaskan resource MediaPipe Face Mesh.")
        self.face_mesh.close()