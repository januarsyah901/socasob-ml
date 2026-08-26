# main.py
"""
main.py — Entry point utama: wiring seluruh komponen sistem.

Alur data:
    Camera → FaceMesh → [vision modules] → [scoring modules]
    → ML Engine → Database + WebSocket broadcast

Arsitektur:
- CV pipeline berjalan di thread sinkron (loop frame-by-frame).
- WebSocket server berjalan di asyncio event loop (thread utama).
- Database ditulis dari pipeline thread (thread-safe via lock).
- Broadcast dari pipeline ke klien WebSocket via run_coroutine_threadsafe().

Kalibrasi baseline:
- Jika CALIBRATE=true (env), sistem akan menjalankan sesi kalibrasi
  ~90 detik di awal untuk mengukur baseline personal blink rate.
- Jika tidak, fallback ke rata-rata populasi (17 bpm) dengan warning.
"""

import asyncio
import os
import signal
import sys
import threading
import time
from typing import Optional

import cv2
import numpy as np

# Camera
from camera.base import CameraSource
from camera.webcam_source import WebcamSource
from camera.esp32_source import ESP32CamSource

# Vision
from cv.face_mesh import FaceMeshDetector
from vision.blink_detector import (
    BlinkEventDetector,
    calculate_ear,
    extract_eye_coordinates,
)
from vision.distance_estimator import DistanceEstimator
from vision.metrics_window import MetricsWindow

# Scoring
from scoring.fatigue_score import calibrate_baseline
from scoring.active_myopia_guard import ActiveMyopiaGuard
from scoring.myopia_risk import MyopiaRiskEstimator

# ML
from ml.engine import InferenceEngine

# Storage & Realtime
from storage.database import HealthDatabase
from realtime.ws_server import RealtimeWSServer
from realtime.hardware_controller import HardwareActuatorController

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
WS_HOST = os.getenv("WS_HOST", "0.0.0.0")
WS_PORT = int(os.getenv("WS_PORT", "8765"))
CALIBRATE = os.getenv("CALIBRATE", "false").lower() == "true"
CALIBRATION_SECONDS = int(os.getenv("CALIBRATION_SECONDS", "90"))
DB_PATH = os.getenv("DB_PATH", None)  # None → default path
# Interval logging ke database (detik) — tidak setiap frame
DB_LOG_INTERVAL = float(os.getenv("DB_LOG_INTERVAL", "5.0"))
# Interval broadcast ke WebSocket (detik)
WS_BROADCAST_INTERVAL = float(os.getenv("WS_BROADCAST_INTERVAL", "1.0"))


# ─────────────────────────────────────────────
# Camera Factory
# ─────────────────────────────────────────────
def create_camera() -> CameraSource:
    """Buat instance kamera berdasarkan konfigurasi."""
    source = getattr(settings, "VIDEO_SOURCE", "webcam").lower()

    if source == "esp32":
        url = getattr(settings, "ESP32_STREAM_URL", "http://192.168.1.100:81/stream")
        logger.info(f"Menggunakan ESP32CamSource: {url}")
        return ESP32CamSource(stream_url=url)
    else:
        index = getattr(settings, "WEBCAM_INDEX", 0)
        logger.info(f"Menggunakan WebcamSource: index={index}")
        return WebcamSource(camera_index=index)


# ─────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────
def run_calibration(
    camera: CameraSource,
    face_mesh: FaceMeshDetector,
    ear_threshold: float,
    duration_sec: int = 90,
) -> float:
    """
    Sesi kalibrasi baseline personal (~90 detik).

    User diminta duduk santai di depan kamera. Sistem menghitung
    blink rate spontan sebagai baseline personal.

    Returns:
        Baseline blink rate (bpm). Fallback 17.0 jika gagal.
    """
    logger.info(f"[Kalibrasi] Memulai sesi kalibrasi {duration_sec} detik...")
    logger.info("[Kalibrasi] Duduk santai di depan kamera. Berkediplah secara alami.")

    detector = BlinkEventDetector(ear_threshold=ear_threshold)
    blink_timestamps = []
    start = time.time()

    while (time.time() - start) < duration_sec:
        frame = camera.read_frame()
        if frame is None:
            time.sleep(0.01)
            continue

        landmarks = face_mesh.process(frame)
        if landmarks is None:
            continue

        h, w = frame.shape[:2]
        left_eye, right_eye = extract_eye_coordinates(landmarks, w, h)
        ear_left = calculate_ear(left_eye)
        ear_right = calculate_ear(right_eye)
        avg_ear = (ear_left + ear_right) / 2.0

        event = detector.update(avg_ear, 1.0, time.time())
        if event:
            blink_timestamps.append(event["timestamp"])

        elapsed = int(time.time() - start)
        remaining = duration_sec - elapsed
        if elapsed % 10 == 0 and remaining > 0:
            logger.info(f"[Kalibrasi] {remaining}s tersisa... ({len(blink_timestamps)} kedipan)")

        time.sleep(0.01)

    baseline = calibrate_baseline(blink_timestamps, duration_sec)
    logger.info(f"[Kalibrasi] Selesai. Baseline personal: {baseline:.1f} bpm "
                f"({len(blink_timestamps)} kedipan dalam {duration_sec}s)")

    if baseline < 5 or baseline > 60:
        logger.warning(f"[Kalibrasi] Baseline {baseline:.1f} bpm di luar range wajar. "
                       "Fallback ke 17 bpm.")
        return 17.0

    return baseline


# ─────────────────────────────────────────────
# Vision Pipeline (thread sinkron)
# ─────────────────────────────────────────────
def vision_pipeline_loop(
    camera: CameraSource,
    face_mesh: FaceMeshDetector,
    blink_detector: BlinkEventDetector,
    distance_estimator: DistanceEstimator,
    metrics_window: MetricsWindow,
    myopia_guard: ActiveMyopiaGuard,
    myopia_risk: MyopiaRiskEstimator,
    engine: InferenceEngine,
    hw_controller: HardwareActuatorController,
    database: HealthDatabase,
    ws_server: RealtimeWSServer,
    event_loop: asyncio.AbstractEventLoop,
    stop_event: threading.Event,
) -> None:
    """
    Loop utama CV pipeline — berjalan di thread terpisah dari asyncio.
    """
    logger.info("[Pipeline] Vision pipeline dimulai.")

    last_db_log = 0.0
    last_ws_broadcast = 0.0
    frame_count = 0

    while not stop_event.is_set():
        frame = camera.read_frame()
        if frame is None:
            time.sleep(0.01)
            continue

        now = time.time()
        frame_count += 1
        h, w = frame.shape[:2]

        # 1. Deteksi landmark wajah
        landmarks = face_mesh.process(frame)
        face_detected = landmarks is not None
        face_confidence = 1.0 if face_detected else 0.0

        avg_ear = 0.0
        distance_cm = None

        if face_detected:
            # 2. Hitung EAR
            left_eye, right_eye = extract_eye_coordinates(landmarks, w, h)
            ear_left = calculate_ear(left_eye)
            ear_right = calculate_ear(right_eye)
            avg_ear = (ear_left + ear_right) / 2.0

            # 3. Estimasi jarak
            distance_cm = distance_estimator.estimate(landmarks, w, h)

        # 4. Update blink detector
        is_closed = (avg_ear < blink_detector.ear_threshold) if face_detected else False
        is_valid = face_confidence >= 0.5

        metrics_window.add_frame(now, is_closed=is_closed, is_valid=is_valid)

        blink_event = blink_detector.update(avg_ear, face_confidence, now)
        if blink_event:
            metrics_window.add_blink(blink_event)

        # 5. Update Modul B1 (jarak + durasi)
        guard_result = myopia_guard.update(
            face_detected=face_detected,
            distance_cm=distance_cm,
            timestamp=now,
        )

        # 6. Update Modul B2 (screen time kumulatif)
        myopia_risk.tick(face_detected)
        risk_result = myopia_risk.get_risk()

        # 7. Jalankan ML Inference Engine
        results = engine.run(
            metrics_window=metrics_window,
            guard_result=guard_result,
            risk_result=risk_result,
        )

        # 8. Evaluasi Perintah Hardware (LCD & Speaker)
        hw_payload = hw_controller.evaluate(results)
        results["hardware"] = hw_payload

        # 9. Log ke database (throttled)
        if (now - last_db_log) >= DB_LOG_INTERVAL:
            try:
                database.log_all(results)
                last_db_log = now
            except Exception as e:
                logger.error(f"[Pipeline] Error logging ke database: {e}")

        # 10. Broadcast ke WebSocket (throttled)
        if (now - last_ws_broadcast) >= WS_BROADCAST_INTERVAL:
            try:
                ws_server.broadcast_results(results, event_loop)
                last_ws_broadcast = now
            except Exception as e:
                logger.debug(f"[Pipeline] Error broadcast WS: {e}")

        # Rate limit pipeline (~30 FPS max)
        time.sleep(0.015)

    logger.info(f"[Pipeline] Dihentikan setelah {frame_count} frame.")


# ─────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────
def main() -> None:
    logger.info("=" * 60)
    logger.info("  SocaSob ML — Sistem Pemantauan Kesehatan Mata")
    logger.info("=" * 60)

    ear_threshold = getattr(settings, "EAR_THRESHOLD", 0.23)

    # 1. Inisialisasi komponen
    logger.info("[Init] Membuat kamera...")
    camera = create_camera()
    if not camera.is_opened():
        logger.error("[Init] Kamera gagal dibuka. Keluar.")
        sys.exit(1)

    frame_w, frame_h = camera.get_resolution()
    logger.info(f"[Init] Kamera siap. Resolusi: {frame_w}x{frame_h}")

    logger.info("[Init] Inisialisasi MediaPipe Face Mesh...")
    face_mesh = FaceMeshDetector()

    logger.info("[Init] Inisialisasi modul vision...")
    blink_detector = BlinkEventDetector(ear_threshold=ear_threshold)
    distance_estimator = DistanceEstimator(frame_width=frame_w)
    metrics_window = MetricsWindow(window_seconds=60)

    logger.info("[Init] Inisialisasi modul scoring...")
    myopia_guard = ActiveMyopiaGuard()
    myopia_risk = MyopiaRiskEstimator()
    myopia_risk.start_session()

    # 2. Kalibrasi baseline (opsional)
    baseline_rate = 17.0
    if CALIBRATE:
        baseline_rate = run_calibration(
            camera, face_mesh, ear_threshold, CALIBRATION_SECONDS
        )
    else:
        logger.warning(
            "[Init] Kalibrasi dilewati. Menggunakan baseline populasi: "
            "17 bpm. Set CALIBRATE=true untuk akurasi personal."
        )

    logger.info("[Init] Inisialisasi ML Inference Engine...")
    engine = InferenceEngine(baseline_rate=baseline_rate)

    logger.info("[Init] Inisialisasi Hardware Actuator Controller...")
    hw_controller = HardwareActuatorController()

    logger.info("[Init] Inisialisasi database...")
    database = HealthDatabase(db_path=DB_PATH)

    logger.info("[Init] Inisialisasi WebSocket server...")
    ws_server = RealtimeWSServer(database=database, host=WS_HOST, port=WS_PORT)

    # 3. Setup asyncio event loop & stop signal
    loop = asyncio.new_event_loop()
    stop_event = threading.Event()

    def shutdown_handler(sig, frame):
        logger.info(f"\n[Shutdown] Sinyal {sig} diterima. Menghentikan...")
        stop_event.set()
        loop.call_soon_threadsafe(loop.stop)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    # 4. Start vision pipeline di thread terpisah
    pipeline_thread = threading.Thread(
        target=vision_pipeline_loop,
        args=(
            camera, face_mesh, blink_detector, distance_estimator,
            metrics_window, myopia_guard, myopia_risk,
            engine, hw_controller, database, ws_server, loop, stop_event,
        ),
        daemon=True,
        name="vision-pipeline",
    )
    pipeline_thread.start()
    logger.info("[Init] Vision pipeline thread dimulai.")

    # 5. Jalankan WebSocket server di asyncio event loop (main thread)
    logger.info(f"[Init] WebSocket server di ws://{WS_HOST}:{WS_PORT}")
    logger.info("=" * 60)
    logger.info("  Sistem siap. Tekan Ctrl+C untuk menghentikan.")
    logger.info("=" * 60)

    try:
        loop.run_until_complete(ws_server.start_async())
    except KeyboardInterrupt:
        pass
    finally:
        # Graceful shutdown
        logger.info("[Shutdown] Membersihkan resource...")
        stop_event.set()
        pipeline_thread.join(timeout=3.0)
        face_mesh.release()
        camera.release()
        loop.run_until_complete(ws_server.stop_async())
        loop.close()
        logger.info("[Shutdown] Selesai. Sampai jumpa!")


if __name__ == "__main__":
    main()
