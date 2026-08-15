# app.py

from flask import Flask, render_template, Response, jsonify, request
import threading
import time
import os

from services.camera_service import CameraService
from services.vision_pipeline_service import VisionPipelineService
from services.feature_store import FeatureStore
from services.stream_service import StreamService
from utils.logger import get_logger

logger = get_logger(__name__)

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# ==========================================
# 1. Inisialisasi Semua Komponen (Services)
# ==========================================
camera_service = CameraService()
pipeline_service = VisionPipelineService(camera_service)
feature_store = FeatureStore()
stream_service = StreamService(pipeline_service)

# ==========================================
# 2. Sinkronisasi Data (Background Worker)
# ==========================================
def sync_features():
    """
    Fungsi ini berjalan terus-menerus di background.
    Tugasnya mengambil data terbaru dari pipeline dan menyimpannya 
    ke FeatureStore agar selalu siap saat API '/api/features' dipanggil.
    """
    while True:
        features, _ = pipeline_service.get_latest_results()
        if features:
            feature_store.update(features)
        time.sleep(0.03) # Jeda ~30ms agar CPU tidak bekerja terlalu keras

# ==========================================
# 3. Menghidupkan Mesin (Lifecycle Hook)
# ==========================================
# Jalankan kamera dan pipeline hanya sekali saat request pertama masuk
@app.route('/health')
def health():
    return jsonify({"status": "ok"}), 200

@app.before_request
def start_background_services():
    if request.path == '/health':
        return
    if not camera_service.is_running:
        logger.info("Menghidupkan Camera dan Vision Pipeline...")
        camera_service.start()
        pipeline_service.start()
        
        # Jalankan sinkronisasi data di thread terpisah
        sync_thread = threading.Thread(target=sync_features, daemon=True)
        sync_thread.start()
        logger.info("Semua background services berhasil dijalankan.")

# ==========================================
# 4. Definisi Endpoint (Routes) HTTP
# ==========================================

@app.route('/')
def index():
    """Endpoint untuk halaman utama (Dashboard HTML)."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Endpoint untuk streaming visual MJPEG."""
    # Mimetype khusus ini memberi tahu browser bahwa ini adalah aliran gambar beruntun (video)
    return Response(
        stream_service.generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/api/features', methods=['GET'])
def api_features():
    """Endpoint API JSON (Kontrak Utama Aplikasi)."""
    # Mengambil data dari cache in-memory yang super cepat
    data = feature_store.get()
    return jsonify(data)

# ==========================================
# 5. Menjalankan Aplikasi
# ==========================================
if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    logger.info("Memulai server SOCACOMVI di http://0.0.0.0:%s", port)
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True, use_reloader=False)