# app.py
#
# Entry point utama ML Server SocaSob.
# Mengorkestrasi seluruh komponen:
#   - Flask-SocketIO sebagai WebSocket SERVER untuk Robot
#   - BackendSocketClient sebagai Socket.io CLIENT ke Backend (Node.js)
#   - VisionPipelineService sebagai mesin CV
#   - AggregatorService sebagai penghitung statistik 1 menit

try:
    import eventlet
    eventlet.monkey_patch()  # Harus sebelum import lain agar async berjalan di Linux/Gunicorn
    ASYNC_MODE = 'eventlet'
except ImportError:
    ASYNC_MODE = 'threading'

from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO

import os
import base64
import numpy as np
import cv2

from services.robot_ws_handler import RobotWebSocketHandler
from services.be_socket_client import BackendSocketClient
from services.aggregator_service import AggregatorService
from services.vision_pipeline_service import VisionPipelineService
from services.feature_store import FeatureStore
from services.stream_service import StreamService
from services.robot_validator import is_robot_registered
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# ==========================================
# 1. Inisialisasi Flask & SocketIO Server
#    (untuk menerima koneksi dari Robot)
# ==========================================
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'socasob-ml-secret')
CORS(app, origins="*")

# SocketIO server di sisi ML — Robot connect ke sini
socketio = SocketIO(app, cors_allowed_origins="*", async_mode=ASYNC_MODE)

try:
    from flask_sock import Sock
    sock = Sock(app)
except ImportError:
    sock = None

from camera.esp32_camera import decode_websocket_packet
from services.robot_trigger_service import RobotTriggerService

# ==========================================
# 2. Inisialisasi Semua Komponen (Services)
# ==========================================
robot_ws_handler = RobotWebSocketHandler(pipeline_service=None)  # pipeline di-set setelah init
be_client = BackendSocketClient()
robot_trigger_service = RobotTriggerService(be_socket_client=be_client)
aggregator = AggregatorService(on_summary=be_client.emit_minute_summary)
pipeline_service = VisionPipelineService(
    robot_ws_handler=robot_ws_handler,
    be_socket_client=be_client,
    aggregator_service=aggregator,
    trigger_service=robot_trigger_service
)

# Set pipeline ke robot_ws_handler (resolve circular dependency)
robot_ws_handler.pipeline = pipeline_service

feature_store = FeatureStore()
stream_service = StreamService(pipeline_service)

# ==========================================
# 3. Background Worker & Services Initialization
# ==========================================
def sync_features():
    """Sinkronisasi hasil pipeline ke FeatureStore untuk endpoint debug."""
    import time
    while True:
        features, _ = pipeline_service.get_latest_results()
        if features:
            feature_store.update(features)
        time.sleep(0.03)  # ~30ms

def start_background_services():
    """Memulai semua thread dan koneksi background (BE socket, aggregator, pipeline CV)."""
    import threading
    logger.info("=== Memulai SocaSob ML Background Services ===")

    # 1. Mulai koneksi ke Backend (async, tidak blocking)
    be_client.connect_async()

    # 2. Mulai aggregator (background timer 60 detik)
    aggregator.start()

    # 3. Mulai vision pipeline (background thread CV)
    pipeline_service.start()

    # 4. Mulai sync feature store (background thread)
    sync_thread = threading.Thread(target=sync_features, daemon=True, name="feature-sync")
    sync_thread.start()

    logger.info(f"ML akan push data ke BE di {settings.BE_URL}")

# Jalankan background services saat module dimuat (kompatibel dengan Gunicorn & direct execution)
start_background_services()

# ==========================================
# 4. WebSocket Event Handlers (Robot → ML)
# ==========================================

@socketio.on('connect')
def on_robot_connect(*args, **kwargs):
    """Dipanggil saat Robot berhasil connect via WebSocket."""
    sid = getattr(request, 'sid', 'unknown')
    logger.info(f"Robot terhubung: sid={sid}")

@socketio.on('disconnect')
def on_robot_disconnect(*args, **kwargs):
    """Dipanggil saat Robot disconnect."""
    sid = getattr(request, 'sid', 'unknown')
    logger.info(f"Robot terputus: sid={sid}")

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

    # Security Gate: Validasi apakah robot terdaftar di sistem
    if not is_robot_registered(robot_id):
        logger.warning(f"[{robot_id}] Frame ditolak: Robot belum terdaftar atau inaktif di sistem.")
        return

    frame_size = len(frame_bytes) if frame_bytes else 0

    # Teruskan ke handler (frame-dropping terjadi di sini)
    robot_ws_handler.on_robot_frame(
        robot_id=robot_id,
        frame_bytes=frame_bytes,
        distance_json=distance_json,
        frame_size_bytes=frame_size
    )

@socketio.on('esp32_frame')
def on_esp32_frame(data):
    """
    Menerima frame binary atau dict dari ESP32-CAM via event esp32_frame.
    """
    frame_bytes = data.get('frame') if isinstance(data, dict) else data
    robot_id = data.get('robot_id') if isinstance(data, dict) else None
    distance_json = data.get('distance_json', {}) if isinstance(data, dict) else {}

    if not robot_id:
        logger.warning("Payload esp32_frame ditolak: robot_id wajib disertakan.")
        return

    if not isinstance(frame_bytes, (bytes, bytearray, memoryview)):
        return

    if not is_robot_registered(robot_id):
        logger.warning(f"[{robot_id}] Frame ditolak: Robot belum terdaftar atau inaktif di sistem.")
        return

    frame_size = len(frame_bytes)
    robot_ws_handler.on_robot_frame(
        robot_id=robot_id,
        frame_bytes=bytes(frame_bytes),
        distance_json=distance_json,
        frame_size_bytes=frame_size
    )

@socketio.on('request_telemetry')
def on_request_telemetry():
    """Mengirim telemetry terakhir ke client Socket.IO."""
    data = feature_store.get()
    if data:
        socketio.emit('telemetry', data)

# ==========================================
# 4b. Raw WebSocket Handlers (ESP32-CAM via /ws)
# ==========================================
if sock is not None:
    def _handle_esp32_raw_websocket(ws):
        logger.info("ESP32-CAM raw WebSocket terhubung.")
        current_robot_id = None
        frame_counter = 0
        try:
            # Kirim handshake response awal begitu terhubung
            try:
                ws.send("READY")
                ws.send("OK")
                ws.send("normal")
                logger.info("Handshake 'READY', 'OK', dan trigger awal 'normal' berhasil dikirim ke ESP32-CAM.")
            except Exception as e:
                logger.warning(f"Gagal kirim handshake awal ke ESP32: {e}")

            while True:
                message = ws.receive()
                if message is None:
                    logger.info("ESP32-CAM ws.receive() returned None (closed).")
                    break

                # Handle jika ESP32 mengirim pesan teks (misal ping / request start / auth)
                if isinstance(message, str):
                    logger.info(f"ESP32-CAM teks diterima: '{message}'")
                    msg_lower = message.lower().strip()
                    if "ping" in msg_lower:
                        try:
                            ws.send("pong")
                        except Exception:
                            pass
                    else:
                        try:
                            ws.send("READY")
                            ws.send("OK")
                        except Exception:
                            pass
                    continue

                if not isinstance(message, (bytes, bytearray, memoryview)):
                    logger.warning(f"ESP32-CAM pesan bukan bytes/str, tipe: {type(message)}")
                    continue

                packet = bytes(message)
                packet_size_bytes = len(packet)
                packet_size_mb = packet_size_bytes / (1024 * 1024)
                packet_size_kb = packet_size_bytes / 1024

                # ----------------------------------------------------------------
                # Format binary ESP32 (wifiStreamTask.cpp):
                #   Byte 0      : robot_id_len (uint8)
                #   Byte 1..N   : robot_id (ASCII, panjang = robot_id_len)
                #   Byte N+1    : is_dekat (0 atau 1)
                #   Byte N+2..  : JPEG frame bytes
                # ----------------------------------------------------------------
                frame = None
                robot_id = None
                is_dekat = False

                try:
                    if len(packet) < 3:
                        raise ValueError("Packet terlalu pendek")

                    robot_id_len = packet[0]
                    if len(packet) < 1 + robot_id_len + 1:
                        raise ValueError(f"Packet terlalu pendek untuk robot_id_len={robot_id_len}")

                    robot_id_raw = packet[1 : 1 + robot_id_len]
                    robot_id = robot_id_raw.decode("ascii", errors="replace").strip("\x00").strip()
                    if not robot_id:
                        raise ValueError("robot_id kosong pada payload binary")

                    is_dekat = bool(packet[1 + robot_id_len])
                    jpeg_bytes = packet[1 + robot_id_len + 1 :]

                    nparr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                except Exception as parse_err:
                    # Fallback ke decode_websocket_packet jika format berbeda
                    decoded = decode_websocket_packet(packet, 16)
                    if decoded is not None:
                        if len(decoded) == 3:
                            robot_id, frame, is_dekat = decoded
                        else:
                            robot_id, frame = decoded
                    else:
                        logger.warning(f"ESP32-CAM gagal decode packet ({len(packet)} bytes): {parse_err}")

                if frame is None or not robot_id:
                    logger.warning(f"ESP32-CAM frame decode gagal ({len(packet)} bytes)")
                    continue

                # Security Gate: Hanya robot yang terdaftar di database Backend yang diproses
                if not is_robot_registered(robot_id):
                    logger.warning(f"[{robot_id}] Frame ditolak: Robot belum terdaftar atau inaktif di sistem.")
                    continue

                current_robot_id = robot_id

                # Daftarkan socket aktif ke trigger service
                robot_trigger_service.register_connection(robot_id, ws)

                frame_counter += 1
                if frame_counter % 30 == 0:
                    logger.info(
                        f"ESP32-CAM streaming aktif: {frame_counter} frame "
                        f"(robot_id={robot_id}, is_dekat={is_dekat}, size={packet_size_mb:.4f} MB, shape={frame.shape})"
                    )

                distance_label = "Dekat" if is_dekat else "Jauh"
                robot_ws_handler.on_frame_array(
                    robot_id=robot_id,
                    frame=frame,
                    distance_json={"distance": distance_label, "confidence": 95},
                    frame_size_bytes=packet_size_bytes
                )
        except Exception as error:
            logger.info(f"ESP32-CAM raw WebSocket terputus setelah {frame_counter} frame: {error}")
        finally:
            if current_robot_id:
                robot_trigger_service.unregister_connection(current_robot_id, ws)


    @sock.route('/ws')
    def handle_esp32_ws(ws):
        _handle_esp32_raw_websocket(ws)

    @sock.route('/ws/esp32')
    def handle_esp32_ws_legacy(ws):
        _handle_esp32_raw_websocket(ws)

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

@app.route('/api/robot/trigger-test', methods=['GET', 'POST'])
def api_trigger_robot_test():
    """
    Endpoint manual untuk menguji pengiriman trigger pesan teks ("normal", "5", "10", "dry") ke robot.
    Menerima parameter 'trigger' dan opsional 'robot_id'.
    """
    if request.method == 'POST':
        data = request.get_json(silent=True) or request.form.to_dict() or {}
        trigger = data.get('trigger')
        robot_id = data.get('robot_id')
    else:
        trigger = request.args.get('trigger')
        robot_id = request.args.get('robot_id')

    if not trigger:
        return jsonify({
            "success": False,
            "error": "Parameter 'trigger' wajib diisi.",
            "supported_triggers": ["normal", "5", "10", "dry"],
            "example_curl": "curl -X POST http://localhost:5000/api/robot/trigger-test -H 'Content-Type: application/json' -d '{\"trigger\": \"dry\", \"robot_id\": \"dummyrobot01\"}'"
        }), 400

    trigger_clean = str(trigger).strip().lower()
    if trigger_clean not in {"normal", "5", "10", "dry"}:
        return jsonify({
            "success": False,
            "error": f"Trigger '{trigger}' tidak valid.",
            "supported_triggers": ["normal", "5", "10", "dry"]
        }), 400

    # Jika robot_id tidak disebutkan, kirim ke semua robot yang terhubung
    if not robot_id:
        count = robot_trigger_service.broadcast_trigger(trigger_clean, force=True)
        return jsonify({
            "success": True,
            "message": f"Trigger '{trigger_clean}' berhasil dibroadcast ke {count} robot terhubung.",
            "trigger": trigger_clean,
            "broadcast_count": count
        }), 200

    sent = robot_trigger_service.send_trigger(robot_id, trigger_clean, force=True)
    return jsonify({
        "success": True,
        "message": f"Trigger '{trigger_clean}' dikirim ke robot '{robot_id}'.",
        "robot_id": robot_id,
        "trigger": trigger_clean,
        "delivered_to_hardware": sent,
        "hardware_connected": robot_trigger_service.is_connected(robot_id)
    }), 200

@app.route('/api/robot/trigger-status', methods=['GET'])
def api_get_robot_trigger_status():
    """Mengecek trigger terakhir dan status koneksi robot."""
    robot_id = request.args.get('robot_id', 'dummyrobot01')
    return jsonify({
        "success": True,
        "robot_id": robot_id,
        "current_trigger": robot_trigger_service.get_last_trigger(robot_id),
        "is_connected": robot_trigger_service.is_connected(robot_id)
    }), 200

@app.route('/api/frame', methods=['POST'])
def api_send_frame():
    """
    HTTP POST Endpoint untuk mengirim frame & status jarak dari Robot/Tester ke ML Server.
    Mendukung JSON payload dan Multipart / Form-Data.
    """
    try:
        robot_id = None
        distance = "Jauh"
        confidence = 90
        frame_bytes = None

        if request.is_json:
            data = request.get_json() or {}
            robot_id = data.get('robot_id')
            distance_json = data.get('distance_json', {})
            distance = distance_json.get('distance', distance)
            confidence = distance_json.get('confidence', confidence)
            
            frame_b64 = data.get('frame_base64') or data.get('frame')
            if frame_b64 and isinstance(frame_b64, str):
                if ',' in frame_b64:
                    frame_b64 = frame_b64.split(',')[1]
                try:
                    frame_bytes = base64.b64decode(frame_b64)
                except Exception as e:
                    logger.warning(f"Failed to decode base64 frame in /api/frame: {e}")
        else:
            robot_id = request.form.get('robot_id')
            distance = request.form.get('distance', distance)
            confidence = int(request.form.get('confidence', confidence))
            if 'frame' in request.files:
                frame_bytes = request.files['frame'].read()

        if not robot_id:
            return jsonify({
                "success": False,
                "error": "Field 'robot_id' wajib disertakan.",
                "hint": "Kirimkan ID robot yang terdaftar di database Backend."
            }), 400

        # Frame wajib ada — tidak boleh dibuat-buat secara sintetis
        if not frame_bytes:
            return jsonify({
                "success": False,
                "error": "Frame gambar wajib dikirim. Gunakan field 'frame' (multipart) atau 'frame_base64' (JSON).",
                "hint": "Kirim file JPEG via multipart/form-data dengan field name 'frame'."
            }), 400

        # Security Gate: Validasi apakah robot terdaftar di sistem database
        if not is_robot_registered(robot_id):
            return jsonify({
                "success": False,
                "error": f"Robot ID '{robot_id}' tidak terdaftar atau sedang inaktif di database Backend. Silakan daftarkan perangkat di Dashboard terlebih dahulu."
            }), 403

        distance_payload = {"distance": distance, "confidence": confidence}

        frame_size = len(frame_bytes)
        frame_mb = round(frame_size / (1024 * 1024), 4)
        frame_kb = round(frame_size / 1024, 2)

        # Teruskan ke robot_ws_handler
        robot_ws_handler.on_robot_frame(
            robot_id=robot_id,
            frame_bytes=frame_bytes,
            distance_json=distance_payload,
            frame_size_bytes=frame_size
        )

        return jsonify({
            "success": True,
            "message": "Frame berhasil diterima dan diproses oleh ML Server",
            "data": {
                "robot_id": robot_id,
                "distance": distance,
                "confidence": confidence,
                "frame_size_bytes": frame_size,
                "frame_size_kb": frame_kb,
                "frame_size_mb": frame_mb,
                "frame_size_formatted": f"{frame_mb:.4f} MB ({frame_kb:.1f} KB)"
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
        robot_id = data.get('robot_id')
        if not robot_id:
            return jsonify({
                "success": False,
                "error": "Field 'robot_id' wajib disertakan."
            }), 400

        if not is_robot_registered(robot_id):
            return jsonify({
                "success": False,
                "error": f"Robot ID '{robot_id}' tidak terdaftar atau inaktif di database Backend."
            }), 403

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
                "flask_host": settings.FLASK_HOST,
                "flask_port": settings.FLASK_PORT,
                "flask_debug": settings.FLASK_DEBUG,
                "video_source": settings.VIDEO_SOURCE,
                "esp32_stream_url": settings.ESP32_STREAM_URL,
                "webcam_index": settings.WEBCAM_INDEX,
                "ear_threshold": settings.EAR_THRESHOLD,
                "consec_frames": settings.CONSEC_FRAMES,
                "log_level": settings.LOG_LEVEL,
                "be_connected": be_client.is_connected,
                "pipeline_running": pipeline_service.is_running if hasattr(pipeline_service, 'is_running') else True
            }
        }), 200
    else:
        data = request.get_json() or {}
        updated = {}

        if 'ear_threshold' in data:
            val = float(data['ear_threshold'])
            if 0.1 <= val <= 0.5:
                settings.EAR_THRESHOLD = val
                updated['ear_threshold'] = val

        if 'consec_frames' in data:
            val = int(data['consec_frames'])
            if 1 <= val <= 10:
                settings.CONSEC_FRAMES = val
                updated['consec_frames'] = val

        if 'video_source' in data:
            if data['video_source'] in ('webcam', 'esp32'):
                settings.VIDEO_SOURCE = data['video_source']
                updated['video_source'] = data['video_source']

        if 'webcam_index' in data:
            settings.WEBCAM_INDEX = int(data['webcam_index'])
            updated['webcam_index'] = settings.WEBCAM_INDEX

        if 'esp32_stream_url' in data:
            settings.ESP32_STREAM_URL = str(data['esp32_stream_url'])
            settings.ESP32_CAM_URL = settings.ESP32_STREAM_URL
            updated['esp32_stream_url'] = settings.ESP32_STREAM_URL

        if 'log_level' in data:
            if data['log_level'] in ('DEBUG', 'INFO', 'WARNING', 'ERROR'):
                settings.LOG_LEVEL = data['log_level']
                updated['log_level'] = data['log_level']

        return jsonify({
            "success": True,
            "message": "Konfigurasi ML Server berhasil diperbarui",
            "updated": updated,
            "data": {
                "ear_threshold": settings.EAR_THRESHOLD,
                "consec_frames": settings.CONSEC_FRAMES,
                "video_source": settings.VIDEO_SOURCE,
                "webcam_index": settings.WEBCAM_INDEX,
                "esp32_stream_url": settings.ESP32_STREAM_URL,
                "log_level": settings.LOG_LEVEL
            }
        }), 200

@app.route('/api/pipeline/status', methods=['GET'])
def api_pipeline_status():
    """
    Ambil status detail pipeline CV.
    """
    data = feature_store.get()
    return jsonify({
        "success": True,
        "data": {
            "be_connected": be_client.is_connected,
            "be_url": settings.BE_URL,
            "last_features": data,
            "has_active_frame": data is not None
        }
    }), 200

# ==========================================
# 6. Entry Point — Menjalankan Server (Direct Run)
# ==========================================
if __name__ == '__main__':
    logger.info("=== Memulai SocaSob ML Server (Direct Execution) ===")
    logger.info(f"Server siap. Robot dapat connect ke ws://0.0.0.0:{settings.FLASK_PORT}")

    # Jalankan Flask-SocketIO server (mendengarkan koneksi dari Robot)
    socketio.run(
        app,
        host=settings.FLASK_HOST,
        port=settings.FLASK_PORT,
        debug=False,
        use_reloader=False,
        allow_unsafe_werkzeug=True
    )