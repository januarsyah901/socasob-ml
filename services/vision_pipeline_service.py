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
from utils.logger import get_logger

logger = get_logger(__name__)


class VisionPipelineService:
    """
    Orkestrasi seluruh langkah Computer Vision (CV) secara berurutan per frame.

    Versi baru: menerima frame dari RobotWebSocketHandler (bukan CameraService),
    dan menghasilkan hasil analisis yang diteruskan ke:
      - FeatureStore (untuk endpoint /api/features debug)
      - BackendSocketClient (Channel A real-time + Channel B via AggregatorService)
    """

    def __init__(self, robot_ws_handler, be_socket_client, aggregator_service):
        """
        Args:
            robot_ws_handler: Instance RobotWebSocketHandler sebagai sumber frame.
            be_socket_client: Instance BackendSocketClient untuk push data ke BE.
            aggregator_service: Instance AggregatorService untuk akumulasi 1 menit.
        """
        self.robot_ws = robot_ws_handler
        self.be_client = be_socket_client
        self.aggregator = aggregator_service

        # Inisialisasi semua modul CV
        self.face_mesh = FaceMeshDetector()
        self.blink_detector = BlinkDetector()
        self.blink_counter = BlinkCounter()
        self.feature_extractor = FeatureExtractor()
        self.visualizer = Visualizer()
        self.fps_counter = FPSCounter()
        self.eye_analyzer = EyeConditionAnalyzer()

        # Shared state untuk hasil akhir (thread-safe) — untuk endpoint debug
        self.latest_features: Dict[str, Any] = {}
        self.latest_annotated_frame: Optional[np.ndarray] = None
        self.lock = threading.Lock()

        self.is_running = False
        self.thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self.is_running:
            return
        self.is_running = True
        self.thread = threading.Thread(
            target=self._processing_loop,
            daemon=True,
            name="vision-pipeline"
        )
        self.thread.start()
        logger.info("VisionPipelineService thread berhasil dimulai.")

    def _processing_loop(self) -> None:
        """
        Loop utama: tunggu frame dari robot, proses, push ke BE.
        Menerapkan frame-dropping secara implisit via RobotWebSocketHandler.
        """
        while self.is_running:
            # Tunggu sampai ada frame baru dari robot (timeout 1 detik)
            has_frame = self.robot_ws.wait_for_frame(timeout=1.0)
            if not has_frame:
                continue

            # Ambil frame + data dari robot
            robot_id, frame, distance_json = self.robot_ws.get_pending()
            if frame is None or robot_id is None:
                continue

            current_time = time.time()
            iso_time = get_current_iso_time()
            fps = self.fps_counter.update()

            # Ambil data jarak dari robot (bukan dihitung di ML)
            distance = distance_json.get("distance", "Jauh")
            confidence = distance_json.get("confidence", 0)

            # 1. Deteksi Wajah & Landmark
            landmarks = self.face_mesh.process(frame)

            blink_event = False
            b_rate = 0.0
            eye_analysis = {"status": "Aman", "conditions": ["Normal"], "recommendations": []}

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
                b_count, b_rate, c_duration = self.blink_counter.process(
                    eye_status, blink_event, current_time
                )

                # 5. Analisis Kondisi Mata dari Blink Rate
                eye_analysis = self.eye_analyzer.analyze_from_blink_rate(b_rate)

                # 6. Build payload untuk feature store (debug endpoint)
                features = self.feature_extractor.build_payload(
                    face_detected=True, timestamp=iso_time, fps=fps,
                    ear=round(avg_ear, 3), eye_status=eye_status,
                    blink_count=b_count, blink_rate=b_rate, closure_duration=c_duration
                )
                features.update({
                    "robot_id": robot_id,
                    "distance": distance,
                    "confidence": confidence,
                    "health_status": eye_analysis["status"],
                    "eye_conditions": eye_analysis["conditions"],
                    "recommendations": eye_analysis["recommendations"]
                })

                # 7. Gambar Visualisasi (untuk endpoint /video_feed debug)
                annotated_frame = self.visualizer.draw_annotations(
                    frame, features, left_eye, right_eye
                )
            else:
                # Fallback jika wajah tidak terdeteksi
                b_count, b_rate, c_duration = 0, 0.0, 0.0
                features = self.feature_extractor.build_payload(
                    face_detected=False, timestamp=iso_time, fps=fps
                )
                features.update({
                    "robot_id": robot_id,
                    "distance": distance,
                    "confidence": confidence,
                })
                annotated_frame = self.visualizer.draw_annotations(frame, features)

            # 8. Simpan hasil ke shared state (thread-safe) untuk debug endpoint
            with self.lock:
                self.latest_features = features
                self.latest_annotated_frame = annotated_frame

            # 9. Push Channel A (real-time) ke BE
            self.be_client.emit_realtime(
                robot_id=robot_id,
                distance=distance,
                confidence=confidence,
                blink_event=blink_event,
                timestamp=iso_time
            )

            # 10. Kirim data ke AggregatorService untuk Channel B (1 menit)
            self.aggregator.ingest(
                robot_id=robot_id,
                distance=distance,
                blink_event=blink_event,
                blink_rate=b_rate,
                health_status=eye_analysis["status"],
                eye_conditions=eye_analysis["conditions"],
                recommendations=eye_analysis["recommendations"]
            )

    def get_latest_results(self) -> Tuple[Dict[str, Any], Optional[np.ndarray]]:
        with self.lock:
            frame_copy = (
                self.latest_annotated_frame.copy()
                if self.latest_annotated_frame is not None
                else None
            )
            return self.latest_features, frame_copy

    def stop(self) -> None:
        logger.info("Menghentikan VisionPipelineService...")
        self.is_running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        self.face_mesh.release()
        logger.info("VisionPipelineService berhasil dihentikan.")