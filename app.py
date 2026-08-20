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

from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO

import os

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
    """Endpoint debug JSON — menampilkan fitur terakhir yang diekstrak."""
    data = feature_store.get()
    return jsonify(data)

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