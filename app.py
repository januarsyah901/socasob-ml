# app.py
#
# Entry point utama ML Server SocaSob.
# Mengorkestrasi seluruh komponen:
#   - Flask-SocketIO sebagai WebSocket SERVER untuk Robot
#   - BackendSocketClient sebagai Socket.io CLIENT ke Backend (Node.js)
#   - VisionPipelineService sebagai mesin CV
#   - AggregatorService sebagai penghitung statistik 1 menit

import eventlet
eventlet.monkey_patch()  # Harus sebelum import lain agar async berjalan

from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO

import os
import base64

from services.robot_ws_handler import RobotWebSocketHandler
from services.be_socket_client import BackendSocketClient
from services.aggregator_service import AggregatorService
from services.vision_pipeline_service import VisionPipelineService
from services.feature_store import FeatureStore
from services.stream_service import StreamService
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# ==========================================
# 1. Inisialisasi Flask & SocketIO Server
#    (untuk menerima koneksi dari Robot)
# ==========================================
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'socasob-ml-secret')

# SocketIO server di sisi ML — Robot connect ke sini
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# ==========================================
# 2. Inisialisasi Semua Komponen (Services)
# ==========================================
robot_ws_handler = RobotWebSocketHandler(pipeline_service=None)  # pipeline di-set setelah init
be_client = BackendSocketClient()
aggregator = AggregatorService(on_summary=be_client.emit_minute_summary)
pipeline_service = VisionPipelineService(
    robot_ws_handler=robot_ws_handler,
    be_socket_client=be_client,
    aggregator_service=aggregator
)

# Set pipeline ke robot_ws_handler (resolve circular dependency)
robot_ws_handler.pipeline = pipeline_service

feature_store = FeatureStore()
stream_service = StreamService(pipeline_service)

# ==========================================
# 3. Background Worker: Sync ke FeatureStore
#    (untuk endpoint /api/features debug)
# ==========================================
def sync_features():
    """Sinkronisasi hasil pipeline ke FeatureStore untuk endpoint debug."""
    import time
    while True:
        features, _ = pipeline_service.get_latest_results()
        if features:
            feature_store.update(features)
        time.sleep(0.03)  # ~30ms

# ==========================================
# 4. WebSocket Event Handlers (Robot → ML)
# ==========================================

@socketio.on('connect')
def on_robot_connect():
    """Dipanggil saat Robot berhasil connect via WebSocket."""
    from flask_socketio import request as ws_request
    logger.info(f"Robot terhubung: sid={ws_request.sid}")

@socketio.on('disconnect')
def on_robot_disconnect():
    """Dipanggil saat Robot disconnect."""
    from flask_socketio import request as ws_request
    logger.info(f"Robot terputus: sid={ws_request.sid}")

@socketio.on('robot-frame')
def on_robot_frame(data):
    """
    Menerima frame dari Robot via WebSocket.

    Payload yang diharapkan dari Robot:
    {
        "robot_id": "fadfa566",
        "frame": <bytes JPEG>,
        "distance_json": {"distance": "Dekat", "confidence": 95}
    }
    """
    robot_id = data.get('robot_id')
    frame_bytes = data.get('frame')
    distance_json = data.get('distance_json', {})

    if not robot_id or not frame_bytes:
        logger.warning("Payload robot-frame tidak valid. Diabaikan.")
        return

    # Teruskan ke handler (frame-dropping terjadi di sini)
    robot_ws_handler.on_robot_frame(
        robot_id=robot_id,
        frame_bytes=frame_bytes,
        distance_json=distance_json
    )

# ==========================================
# 5. HTTP Routes (Debug & Health)
# ==========================================

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "be_connected": be_client.is_connected
    }), 200

@app.route('/')
def index():
    """Halaman debug visual (opsional)."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """MJPEG stream debug — menampilkan frame terakhir yang diproses."""
    return Response(
        stream_service.generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/api/features', methods=['GET'])
def api_features():
    """
    Endpoint debug JSON — menampilkan fitur terakhir yang diekstrak dari pipeline CV.
    Mengembalikan 404 jika belum ada robot yang connect dan mengirim frame.
    """
    data = feature_store.get()
    if data is None:
        return jsonify({
            "success": False,
            "error": "Belum ada data fitur. Robot belum connect atau belum ada frame yang diproses.",
            "hint": "Kirim frame via WebSocket (robot-frame) atau POST /api/frame terlebih dahulu."
        }), 404
    return jsonify({"success": True, "data": data})

@app.route('/api/frame', methods=['POST'])
def api_send_frame():
    """
    HTTP POST Endpoint untuk mengirim frame & status jarak dari Robot/Tester ke ML Server.
    Mendukung JSON payload dan Multipart / Form-Data.
    """
    try:
        robot_id = "fadfa566"
        distance = "Jauh"
        confidence = 90
        frame_bytes = None

        if request.is_json:
            data = request.get_json() or {}
            robot_id = data.get('robot_id', robot_id)
            distance_json = data.get('distance_json', {})
            distance = distance_json.get('distance', distance)
            confidence = distance_json.get('confidence', confidence)
            
            frame_b64 = data.get('frame_base64')
            if frame_b64:
                if ',' in frame_b64:
                    frame_b64 = frame_b64.split(',')[1]
                frame_bytes = base64.b64decode(frame_b64)
        else:
            robot_id = request.form.get('robot_id', robot_id)
            distance = request.form.get('distance', distance)
            confidence = int(request.form.get('confidence', confidence))
            if 'frame' in request.files:
                frame_bytes = request.files['frame'].read()

        # Jika tidak ada frame binary, buat synthetic dummy frame untuk kemudahan testing
        if not frame_bytes:
            import numpy as np
            import cv2
            img = np.zeros((240, 320, 3), dtype=np.uint8)
            color = (60, 60, 200) if distance == "Dekat" else (200, 100, 60)
            img[:] = color
            cv2.putText(img, distance.upper(), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            _, buf = cv2.imencode('.jpg', img)
            frame_bytes = buf.tobytes()

        distance_payload = {"distance": distance, "confidence": confidence}

        # Teruskan ke robot_ws_handler
        robot_ws_handler.on_robot_frame(
            robot_id=robot_id,
            frame_bytes=frame_bytes,
            distance_json=distance_payload
        )

        return jsonify({
            "success": True,
            "message": "Frame berhasil diterima dan diproses oleh ML Server",
            "data": {
                "robot_id": robot_id,
                "distance": distance,
                "confidence": confidence,
                "frame_size_bytes": len(frame_bytes)
            }
        }), 200
    except Exception as e:
        logger.error(f"Error pada POST /api/frame: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/summary/trigger', methods=['POST'])
def api_trigger_summary():
    """
    HTTP POST Endpoint untuk memicu / memproses ringkasan 1 menit secara manual.
    Berguna untuk pengujian integrasi ML -> BE Channel B via Bruno/cURL.
    """
    try:
        data = request.get_json() or {}
        robot_id = data.get('robot_id', 'fadfa566')
        near_sec = data.get('near_duration_sec', 40)
        far_sec = data.get('far_duration_sec', 20)
        blink_count = data.get('blink_count', 12)

        summary_payload = {
            "robot_id": robot_id,
            "period_start": data.get('period_start', "2026-08-22T18:00:00+07:00"),
            "period_end": data.get('period_end', "2026-08-22T18:01:00+07:00"),
            "near_duration_sec": near_sec,
            "far_duration_sec": far_sec,
            "near_percentage": round((near_sec / (near_sec + far_sec)) * 100, 1) if (near_sec + far_sec) > 0 else 0,
            "blink_count": blink_count,
            "avg_blink_rate": round(blink_count / 1.0, 1),
            "dominant_distance": "Dekat" if near_sec >= far_sec else "Jauh",
            "health_status": "Peringatan" if near_sec > far_sec else "Aman",
            "eye_conditions": ["Resiko Kelelahan Mata"],
            "recommendations": ["Istirahat 20 detik."]
        }

        # Emit langsung ke BE Socket client
        be_client.emit_minute_summary(summary_payload)

        return jsonify({
            "success": True,
            "message": "Manual minute summary berhasil dikirim ke Backend",
            "data": summary_payload
        }), 200
    except Exception as e:
        logger.error(f"Error pada POST /api/summary/trigger: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/config', methods=['GET', 'POST'])
def api_config():
    """
    GET: Ambil konfigurasi aktif ML Server.
    POST: Perbarui parameter konfigurasi ML Server secara dinamis.
    """
    if request.method == 'GET':
        return jsonify({
            "success": True,
            "data": {
                "be_url": settings.BE_URL,
                "flask_port": settings.FLASK_PORT,
                "ear_threshold": getattr(settings, 'EAR_THRESHOLD', 0.21),
                "be_connected": be_client.is_connected
            }
        }), 200
    else:
        data = request.get_json() or {}
        if 'ear_threshold' in data:
            settings.EAR_THRESHOLD = float(data['ear_threshold'])
        return jsonify({
            "success": True,
            "message": "Konfigurasi ML Server berhasil diperbarui",
            "data": {
                "ear_threshold": getattr(settings, 'EAR_THRESHOLD', 0.21)
            }
        }), 200

# ==========================================
# 6. Entry Point — Menjalankan Semua Service
# ==========================================
if __name__ == '__main__':
    logger.info("=== Memulai SocaSob ML Server ===")

    # Mulai koneksi ke Backend (async, tidak blocking)
    be_client.connect_async()

    # Mulai aggregator (background timer 60 detik)
    aggregator.start()

    # Mulai vision pipeline (background thread CV)
    pipeline_service.start()

    # Mulai sync feature store (background thread)
    import threading
    sync_thread = threading.Thread(target=sync_features, daemon=True, name="feature-sync")
    sync_thread.start()

    logger.info(f"Server siap. Robot dapat connect ke ws://0.0.0.0:{settings.FLASK_PORT}")
    logger.info(f"ML akan push data ke BE di {settings.BE_URL}")

    # Jalankan Flask-SocketIO server (mendengarkan koneksi dari Robot)
    socketio.run(
        app,
        host=settings.FLASK_HOST,
        port=settings.FLASK_PORT,
        debug=False,
        use_reloader=False
    )