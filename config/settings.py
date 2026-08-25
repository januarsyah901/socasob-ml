# config/settings.py

import os

# ==========================================
# Flask Server Configuration
# ==========================================
FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("PORT", os.getenv("FLASK_PORT", 5000)))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "True").lower() == "true"

# ==========================================
# Video Source Configuration
# ==========================================
# Opsi: 'webcam' atau 'esp32'
VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", "webcam")

# URL untuk ESP32-CAM (Jika VIDEO_SOURCE = 'esp32')
ESP32_STREAM_URL = os.getenv("ESP32_STREAM_URL", "http://192.168.1.100:81/stream")
ESP32_CAM_URL = os.getenv("ESP32_CAM_URL", ESP32_STREAM_URL)

# Index untuk Webcam (Jika VIDEO_SOURCE = 'webcam')
WEBCAM_INDEX = int(os.getenv("WEBCAM_INDEX", 0))

# ==========================================
# Computer Vision / Eye Health Parameters
# ==========================================
# Threshold untuk Eye Aspect Ratio (EAR) yang dianggap "mata tertutup"
EAR_THRESHOLD = float(os.getenv("EAR_THRESHOLD", 0.20))

# Jumlah frame berturut-turut di mana EAR harus di bawah threshold 
# untuk dianggap sebagai kedipan (blink) yang valid
CONSEC_FRAMES = int(os.getenv("CONSEC_FRAMES", 3))

# ==========================================
# Backend Socket.io Connection
# ==========================================
# URL Backend Node.js yang akan menerima data dari ML
BE_URL = os.getenv("BE_URL", "http://localhost:3001")

# ==========================================
# Logging Configuration
# ==========================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")