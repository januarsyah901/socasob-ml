# services/vision_pipeline_service.py

import threading
import time
import numpy as np
from typing import Optional, Dict, Any, Tuple

from cv.face_mesh import FaceMeshDetector
from cv.eye_landmarks import extract_eye_coordinates
from cv.ear import calculate_ear
from cv.eye_analyzer import EyeConditionAnalyzer
from cv.feature_extractor import FeatureExtractor
from cv.visualization import Visualizer
from utils.fps_counter import FPSCounter
from utils.time_utils import get_current_iso_time
from utils.logger import get_logger
from config import settings

from realtime.hardware_controller import HardwareActuatorController

logger = get_logger(__name__)


class VisionPipelineService:
    """
    Orkestrasi seluruh langkah Computer Vision (CV) per-frame:
    - Ekstraksi landmark MediaPipe & kalkulasi EAR
    - Scoring komposit & klasifikasi fatigue (EyeConditionAnalyzer)
    - Pembangunan payload terstandar (FeatureExtractor)
    - Visualisasi anotasi frame untuk stream debug (Visualizer)
    - Distribusi data ke Backend (Channel A real-time & Channel B agregasi)
    """

    def __init__(self, robot_ws_handler, be_socket_client, aggregator_service, trigger_service=None):
        """
        Args:
            robot_ws_handler: Instance RobotWebSocketHandler sebagai sumber frame.
            be_socket_client: Instance BackendSocketClient untuk push data ke BE.
            aggregator_service: Instance AggregatorService untuk akumulasi 1 menit.
            trigger_service: Instance RobotTriggerService untuk trigger teks ke ESP32.
        """
        self.robot_ws = robot_ws_handler
        self.be_client = be_socket_client
        self.aggregator = aggregator_service
        self.trigger_service = trigger_service

        # Inisialisasi modul-modul CV
        self.face_mesh = FaceMeshDetector()
        self.eye_analyzer = EyeConditionAnalyzer(
            ear_threshold=getattr(settings, 'EAR_THRESHOLD', 0.23),
            window_seconds=60,
            required_consecutive=3,
            min_data_quality=0.7,
            baseline_rate=17.0
        )
        self.feature_extractor = FeatureExtractor()
        self.visualizer = Visualizer()
        self.fps_counter = FPSCounter()
        self.hw_controller = HardwareActuatorController()

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
        Loop utama: tunggu frame dari robot, proses analitik CV, push ke BE & Aggregator.
        """
        while self.is_running:
            has_frame = self.robot_ws.wait_for_frame(timeout=1.0)
            if not has_frame:
                continue

            pending_res = self.robot_ws.get_pending()
            if len(pending_res) == 4:
                robot_id, frame, distance_json, frame_size_bytes = pending_res
            else:
                robot_id, frame, distance_json = pending_res
                frame_size_bytes = frame.nbytes if frame is not None else 0

            if frame is None or robot_id is None:
                continue

            frame_size_bytes = frame_size_bytes or (frame.nbytes if frame is not None else 0)
            frame_size_mb = round(frame_size_bytes / (1024 * 1024), 4)
            frame_size_kb = round(frame_size_bytes / 1024, 2)

            current_time = time.time()
            iso_time = get_current_iso_time()
            fps = self.fps_counter.update()

            distance = distance_json.get("distance", "Jauh")
            confidence = distance_json.get("confidence", 0)

            # 1. Deteksi Wajah & Landmark
            landmarks = self.face_mesh.process(frame)
            face_detected = (landmarks is not None)
            face_confidence = 1.0 if face_detected else 0.0

            left_eye, right_eye = None, None
            avg_ear = 0.0
            eye_status = "Unknown"

            if face_detected and landmarks:
                h, w = frame.shape[:2]

                # 2. Ekstrak Landmark Mata
                left_eye, right_eye = extract_eye_coordinates(landmarks, w, h)

                # 3. Hitung EAR (Eye Aspect Ratio)
                ear_left = calculate_ear(left_eye)
                ear_right = calculate_ear(right_eye)
                avg_ear = (ear_left + ear_right) / 2.0
                eye_status = "Closed" if avg_ear < self.eye_analyzer.detector.ear_threshold else "Open"

            # 4. Evaluasi Scoring Komposit Mata Lelah & Kering
            blink_event, metrics_dict = self.eye_analyzer.process_frame(
                ear_value=avg_ear,
                face_confidence=face_confidence,
                timestamp=current_time
            )

            # 5. Bangun payload fitur lengkap
            features = self.feature_extractor.build_payload(
                face_detected=face_detected,
                timestamp=iso_time,
                fps=fps,
                ear=round(avg_ear, 3) if face_detected else 0.0,
                eye_status=eye_status,
                blink_count=metrics_dict["blink_count"],
                lifetime_blinks=metrics_dict["lifetime_blinks"],
                blink_rate=metrics_dict["smoothed_blink_rate"],
                raw_blink_rate=metrics_dict["raw_blink_rate"],
                closure_duration=metrics_dict["avg_blink_duration"],
                perclos=metrics_dict["perclos"],
                composite_score=metrics_dict["composite_score"],
                avg_blink_duration=metrics_dict["avg_blink_duration"],
                interval_variability=metrics_dict["interval_variability"],
                data_quality=metrics_dict["data_quality"],
                system_status=metrics_dict["system_status"]
            )

            # Evaluasi Hardware Command (LCD & Speaker)
            eval_dict = {
                "fatigue": {"status": metrics_dict["system_status"]},
                "dry_eye": {"status": metrics_dict["system_status"] if "Ringan" in metrics_dict["system_status"] or "Kritis" in metrics_dict["system_status"] else "Aman"},
                "myopia_risk": {"break_state": "active", "break_remaining_sec": 0.0}
            }
            hw_payload = self.hw_controller.evaluate(eval_dict)
            trigger_text = hw_payload.get("robot_trigger", "normal")

            # Kirim trigger pesan teks ke robot jika trigger service aktif
            if self.trigger_service is not None and robot_id:
                self.trigger_service.send_trigger(robot_id, trigger_text)

            features.update({
                "robot_id": robot_id,
                "frame_size_bytes": frame_size_bytes,
                "frame_size_kb": frame_size_kb,
                "frame_size_mb": frame_size_mb,
                "frame_size_formatted": f"{frame_size_mb:.4f} MB ({frame_size_kb:.1f} KB)",
                "distance": distance,
                "confidence": confidence,
                "health_status": metrics_dict["health_status"],
                "eye_conditions": metrics_dict["conditions"],
                "recommendations": metrics_dict["recommendations"],
                "hardware": hw_payload,
                "robot_trigger": trigger_text,
                "work_elapsed_sec": hw_payload.get("work_elapsed_sec", 0),
                "break_remaining_sec": hw_payload.get("break_remaining_sec", 0)
            })

            # 6. Gambar Visualisasi Anotasi
            annotated_frame = self.visualizer.draw_annotations(
                frame=frame,
                features=features,
                left_eye=left_eye,
                right_eye=right_eye
            )

            # 7. Simpan hasil ke shared state (thread-safe) untuk endpoint debug
            with self.lock:
                self.latest_features = features
                self.latest_annotated_frame = annotated_frame

            # 8. Push Channel A (real-time) ke BE
            self.be_client.emit_realtime(
                robot_id=robot_id,
                distance=distance,
                confidence=confidence,
                blink_event=blink_event,
                timestamp=iso_time
            )

            # 9. Kirim data ke AggregatorService untuk Channel B (1 menit)
            self.aggregator.ingest(
                robot_id=robot_id,
                distance=distance,
                blink_event=blink_event,
                blink_rate=metrics_dict["smoothed_blink_rate"],
                health_status=metrics_dict["health_status"],
                eye_conditions=metrics_dict["conditions"],
                recommendations=metrics_dict["recommendations"],
                perclos=metrics_dict["perclos"],
                composite_score=metrics_dict["composite_score"]
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