# services/vision_pipeline_service.py

import threading
import time
import numpy as np
import cv2
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

from realtime.daily_hardware_policy import DailyHardwarePolicy
from vision.distance_estimator import DistanceEstimator

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
            ear_threshold=0.21,
            window_seconds=60,
            required_consecutive=3,
            min_data_quality=0.7,
            baseline_rate=17.0
        )
        self.feature_extractor = FeatureExtractor()
        self.visualizer = Visualizer()
        self.fps_counter = FPSCounter()
        self.daily_policy = DailyHardwarePolicy()
        self.distance_estimator = DistanceEstimator()

        # Shared state untuk hasil akhir (thread-safe) — untuk endpoint debug
        self.latest_features: Dict[str, Any] = {}
        self.latest_annotated_frame: Optional[np.ndarray] = None
        self.frames_by_robot: Dict[str, np.ndarray] = {}
        self.features_by_robot: Dict[str, Dict[str, Any]] = {}
        self.last_frame_time_by_robot: Dict[str, float] = {}
        self.lock = threading.Lock()

        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        self.last_frame_time = 0.0
        self.last_robot_id: Optional[str] = None

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
        logger.info("Vision pipeline loop berjalan...")

        while self.is_running:
            try:
                has_frame = self.robot_ws.wait_for_frame(timeout=1.0)
                if not has_frame:
                    continue

                pending_res = self.robot_ws.get_pending()
                if len(pending_res) == 4:
                    robot_id, frame, distance_json, frame_size_bytes = pending_res
                else:
                    robot_id, frame, distance_json = pending_res
                    frame_size_bytes = 0

                if frame is None or robot_id is None:
                    continue

                frame = self.face_mesh.preprocess_frame(frame)

                frame_size_bytes = frame_size_bytes or (frame.nbytes if frame is not None else 0)
                frame_size_mb = round(frame_size_bytes / (1024 * 1024), 4)
                frame_size_kb = round(frame_size_bytes / 1024, 2)

                current_time = time.time()
                self.last_frame_time = current_time
                self.last_robot_id = robot_id
                iso_time = get_current_iso_time()
                fps = self.fps_counter.update()

                distance = distance_json.get("distance", "Tidak diketahui")
                confidence = distance_json.get("confidence", 0)

                # 1. Deteksi Wajah & Landmark
                landmarks = self.face_mesh.process_preprocessed(frame)
                face_detected = (landmarks is not None)
                face_confidence = 1.0 if face_detected else 0.0

                left_eye, right_eye = None, None
                avg_ear = 0.0
                eye_status = "Unknown"
                estimated_distance_cm = None

                if face_detected and landmarks:
                    h, w = frame.shape[:2]
                    left_eye, right_eye = extract_eye_coordinates(landmarks, w, h)
                    ear_left = calculate_ear(left_eye)
                    ear_right = calculate_ear(right_eye)
                    avg_ear = (ear_left + ear_right) / 2.0
                    eye_status = "Unknown"
                    estimated_distance_cm = self.distance_estimator.estimate(landmarks, w, h)

                # Prioritas distance_cm: dari sensor robot jika ada, atau estimasi CV vision
                distance_cm = distance_json.get("distance_cm")
                if distance_cm is None and "distance_mm" in distance_json:
                    try:
                        distance_cm = round(float(distance_json["distance_mm"]) / 10.0, 1)
                    except (ValueError, TypeError):
                        pass
                if distance_cm is None:
                    distance_cm = estimated_distance_cm

                # Update jarak jika estimasi tersedia dan sensor robot belum kirim status eksplisit
                if distance_cm is not None and "distance" not in distance_json:
                    distance = "Dekat" if distance_cm < 30.0 else "Jauh"

                # 2. Analisis Kondisi Mata
                blink_event, metrics_dict = self.eye_analyzer.process_frame(
                    ear_value=avg_ear,
                    face_confidence=face_confidence,
                    timestamp=current_time
                )
                smoothed_ear = avg_ear
                eye_status = (
                    "Closed" if self.eye_analyzer.detector.state.value == "closed" else "Open"
                ) if face_detected else "Unknown"

                # 3. Ekstraksi Payload Fitur Terstandar
                features = self.feature_extractor.build_payload(
                    face_detected=face_detected,
                    timestamp=iso_time,
                    fps=fps,
                    ear=round(smoothed_ear, 3) if face_detected else 0.0,
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

                # 4. ML policy adalah satu-satunya pembuat command hardware.
                risk_ready = metrics_dict.get("warmup_complete", False)
                incomplete_blink = bool(risk_ready and blink_event and metrics_dict.get("incomplete", False))
                hw_policy_results = self.daily_policy.update(
                    robot_id=robot_id,
                    face_detected=face_detected,
                    distance_cm=distance_cm,
                    blink_event=bool(risk_ready and blink_event),
                    incomplete_blink=incomplete_blink,
                    risk_ready=risk_ready,
                    valid_observation_time=self.eye_analyzer.window.valid_observation_time(),
                )
                hw_payload = hw_policy_results
                trigger_text = hw_payload["hardware_command"]

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
                    "distance_cm": distance_cm,
                    "confidence": confidence,
                    "health_status": metrics_dict["health_status"],
                    "eye_conditions": metrics_dict["conditions"],
                    "recommendations": metrics_dict["recommendations"],
                    "hardware": hw_payload,
                    "robot_trigger": trigger_text,
                    "hardware_command": trigger_text,
                    "screen_time_minutes": hw_payload["screen_time_minutes"],
                    "continuous_gaze_minutes": hw_payload["continuous_gaze_minutes"],
                    "close_distance_duration_seconds": hw_payload["close_distance_duration_seconds"],
                    "blink_rate_per_minute": hw_payload["blink_rate_per_minute"],
                    "incomplete_blink_count": hw_payload["incomplete_blink_count"],
                    "total_blink_observed": hw_payload["total_blink_observed"],
                    "incomplete_blink_ratio": hw_payload["incomplete_blink_ratio"],
                    "fatigue_risk": hw_payload["fatigue_risk"],
                    "dry_eye_risk": hw_payload["dry_eye_risk"],
                    "myopia_report_risk": hw_payload["myopia_report_risk"],
                    "work_elapsed_sec": hw_payload.get("work_elapsed_sec", 0),
                    "break_remaining_sec": hw_payload.get("break_remaining_sec", 0)
                })

                # 5. Gambar Visualisasi Anotasi
                annotated_frame = self.visualizer.draw_annotations(
                    frame=frame,
                    features=features,
                    left_eye=left_eye,
                    right_eye=right_eye
                )

                # 6. Simpan hasil ke shared state (thread-safe) untuk endpoint debug
                with self.lock:
                    self.latest_features = features
                    self.latest_annotated_frame = annotated_frame
                    if robot_id:
                        self.frames_by_robot[robot_id] = annotated_frame
                        self.features_by_robot[robot_id] = features
                        self.last_frame_time_by_robot[robot_id] = current_time


                # 7. Push Channel A (real-time) ke BE
                self.be_client.emit_realtime(
                    robot_id=robot_id,
                    distance=distance,
                    confidence=confidence,
                    blink_event=blink_event,
                    timestamp=iso_time
                )

                # 8. Kirim data ke AggregatorService untuk Channel B (1 menit)
                self.aggregator.ingest(
                    robot_id=robot_id,
                    face_detected=face_detected,
                    distance_cm=distance_cm,
                    blink_event=bool(risk_ready and blink_event),
                    incomplete_blink=incomplete_blink,
                    policy_summary=hw_payload,
                )
            except Exception as e:
                logger.error(f"Error pada vision pipeline loop: {e}", exc_info=True)
                time.sleep(0.01)

    def get_latest_results(self, robot_id: Optional[str] = None) -> Tuple[Dict[str, Any], Optional[np.ndarray]]:
        with self.lock:
            if robot_id and robot_id in self.frames_by_robot:
                feat = self.features_by_robot.get(robot_id, self.latest_features)
                ann = self.frames_by_robot[robot_id]
                last_time = self.last_frame_time_by_robot.get(robot_id, self.last_frame_time)
            else:
                feat = self.latest_features
                ann = self.latest_annotated_frame
                last_time = self.last_frame_time

            if ann is None:
                return feat, None
            
            frame_copy = ann.copy()
            # Jika tidak ada frame baru selama > 3.0 detik, tampilkan status offline/waiting overlay
            if time.time() - last_time > 3.0 and frame_copy is not None:
                h, w = frame_copy.shape[:2]
                cv2.rectangle(frame_copy, (0, h // 2 - 30), (w, h // 2 + 30), (0, 0, 150), -1)
                lbl = f"ESP32-CAM ({robot_id or 'ROBOT'}) OFFLINE / WAITING..."
                cv2.putText(
                    frame_copy,
                    lbl,
                    (20, h // 2 + 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )
            return feat, frame_copy

    def get_all_results(self) -> Dict[str, Dict[str, Any]]:
        """
        Mengembalikan dict robot_id -> features untuk semua robot yang pernah diproses.
        """
        with self.lock:
            res = {k: v.copy() for k, v in self.features_by_robot.items() if v}
            if not res and self.latest_features:
                rid = self.latest_features.get("robot_id") or "default"
                res[rid] = self.latest_features.copy()
            return res

    def stop(self) -> None:
        logger.info("Menghentikan VisionPipelineService...")
        self.is_running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        self.face_mesh.release()
        logger.info("VisionPipelineService berhasil dihentikan.")

    def list_robots(self) -> list:
        """
        Daftar robot yang aktif di pipeline.
        """
        now = time.time()
        with self.lock:
            result = []
            for rid, last_t in self.last_frame_time_by_robot.items():
                is_active = (now - last_t) < 30.0
                result.append({
                    "robot_id": rid,
                    "last_seen": last_t,
                    "is_active": is_active
                })
            if not result and self.last_robot_id:
                is_active = (now - self.last_frame_time) < 30.0
                result.append({
                    "robot_id": self.last_robot_id,
                    "last_seen": self.last_frame_time,
                    "is_active": is_active
                })
            return result

    def set_ear_threshold(self, threshold: float) -> int:
        """
        Memperbarui batas threshold EAR pada analyzer mata.
        """
        with self.lock:
            if hasattr(self, 'eye_analyzer') and self.eye_analyzer:
                self.eye_analyzer.threshold = threshold
                self.eye_analyzer.detector.ear_threshold = threshold
                self.eye_analyzer.detector.close_threshold = threshold
                return 1
            return 0

    def reset(self, robot_id: Optional[str] = None) -> None:
        """Reset analitik vision pipeline, hardware controller, dan cache fitur."""
        with self.lock:
            if robot_id:
                self.frames_by_robot.pop(robot_id, None)
                self.features_by_robot.pop(robot_id, None)
                self.last_frame_time_by_robot.pop(robot_id, None)
                if self.last_robot_id == robot_id:
                    self.last_robot_id = None
                    self.latest_features = {}
                    self.latest_annotated_frame = None
            else:
                self.latest_features = {}
                self.latest_annotated_frame = None
                self.frames_by_robot.clear()
                self.features_by_robot.clear()
                self.last_frame_time_by_robot.clear()
                self.last_robot_id = None

        if hasattr(self, 'eye_analyzer') and self.eye_analyzer:
            self.eye_analyzer.reset()

        if hasattr(self, 'fps_counter') and self.fps_counter:
            self.fps_counter.reset()
            
        if hasattr(self, 'daily_policy') and self.daily_policy:
            self.daily_policy.reset(robot_id)

        logger.info(f"[VisionPipeline] Pipeline direset ke kondisi awal untuk '{robot_id or 'all'}'.")