# services/vision_pipeline_service.py

from cv.eye_analyzer import EyeConditionAnalyzer

import threading
import time
import numpy as np
from typing import Optional, Dict, Any, Tuple

from cv.face_mesh import FaceMeshDetector
from cv.eye_landmarks import extract_eye_coordinates
from cv.ear import calculate_ear
from cv.blink_detector import BlinkDetector
from cv.blink_counter import BlinkCounter
from cv.feature_extractor import FeatureExtractor
from cv.visualization import Visualizer
from utils.fps_counter import FPSCounter
from utils.time_utils import get_current_iso_time
from services.camera_service import CameraService
from utils.logger import get_logger

logger = get_logger(__name__)

class VisionPipelineService:
    """
    Orkestrasi seluruh langkah Computer Vision (CV) secara berurutan per frame.
    """

    def __init__(self, camera_service: CameraService):
        self.camera = camera_service
        
        # Inisialisasi semua modul CV
        self.face_mesh = FaceMeshDetector()
        self.blink_detector = BlinkDetector()
        self.blink_counter = BlinkCounter()
        self.feature_extractor = FeatureExtractor()
        self.visualizer = Visualizer()
        self.fps_counter = FPSCounter()
        
        # --- 1. INISIALISASI ANALYZER ---
        self.eye_analyzer = EyeConditionAnalyzer()

        # Shared state untuk hasil akhir (thread-safe)
        self.latest_features: Dict[str, Any] = {}
        self.latest_annotated_frame: Optional[np.ndarray] = None
        
        self.lock = threading.Lock()
        self.is_running = False
        self.thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self.is_running:
            return
        
        self.is_running = True
        self.thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.thread.start()
        logger.info("VisionPipelineService thread berhasil dimulai.")

    def _processing_loop(self) -> None:
        while self.is_running:
            frame = self.camera.get_latest_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            current_time = time.time()
            iso_time = get_current_iso_time()
            fps = self.fps_counter.update()

            # 1. Deteksi Wajah & Landmark
            landmarks = self.face_mesh.process(frame)

            if landmarks:
                h, w = frame.shape[:2]
                
                # 2. Ekstrak Mata
                left_eye, right_eye = extract_eye_coordinates(landmarks, w, h)
                
                # 3. Hitung EAR Rata-rata
                ear_left = calculate_ear(left_eye)
                ear_right = calculate_ear(right_eye)
                avg_ear = (ear_left + ear_right) / 2.0
                
                # 4. Deteksi Kedipan & Hitung Statistik
                eye_status, blink_event = self.blink_detector.process(avg_ear)
                b_count, b_rate, c_duration = self.blink_counter.process(eye_status, blink_event, current_time)
                
                # --- 2. JALANKAN ANALISIS KONDISI MATA DARI BLINK RATE ---
                eye_analysis = self.eye_analyzer.analyze_from_blink_rate(b_rate)
                
                # (Opsional) Jika nanti sudah ada fitur deteksi jarak/waktu, masukkan ke sini
                # myopia_analysis = self.eye_analyzer.analyze_myopia_risk(distance_cm=..., screen_time_hours=...)

                # 5. Gabungkan menjadi Payload JSON
                features = self.feature_extractor.build_payload(
                    face_detected=True, timestamp=iso_time, fps=fps,
                    ear=round(avg_ear, 3), eye_status=eye_status,
                    blink_count=b_count, blink_rate=b_rate, closure_duration=c_duration
                )
                
                # --- 3. SUNTIKKAN HASIL ANALISIS KE DALAM PAYLOAD JSON ---
                features.update({
                    "health_status": eye_analysis["status"],
                    "eye_conditions": eye_analysis["conditions"],
                    "recommendations": eye_analysis["recommendations"]
                })
                
                # 6. Gambar Visualisasi
                annotated_frame = self.visualizer.draw_annotations(frame, features, left_eye, right_eye)
            else:
                # Fallback jika wajah tidak terdeteksi
                features = self.feature_extractor.build_payload(
                    face_detected=False, timestamp=iso_time, fps=fps
                )
                annotated_frame = self.visualizer.draw_annotations(frame, features)

            # Simpan hasil akhir secara thread-safe
            with self.lock:
                self.latest_features = features
                self.latest_annotated_frame = annotated_frame

    def get_latest_results(self) -> Tuple[Dict[str, Any], Optional[np.ndarray]]:
        with self.lock:
            frame_copy = self.latest_annotated_frame.copy() if self.latest_annotated_frame is not None else None
            return self.latest_features, frame_copy

    def stop(self) -> None:
        logger.info("Menghentikan VisionPipelineService...")
        self.is_running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        self.face_mesh.release()
        logger.info("VisionPipelineService berhasil dihentikan.")