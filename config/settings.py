import os
from pathlib import Path

# Load .env file automatically if exists
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip())

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
EAR_THRESHOLD = float(os.getenv("EAR_THRESHOLD", 0.23))

# Jumlah frame berturut-turut di mana EAR harus di bawah threshold 
# untuk dianggap sebagai kedipan (blink) yang valid (1-2 frame untuk responsive detection)
CONSEC_FRAMES = int(os.getenv("CONSEC_FRAMES", 1))

# ==========================================
# Backend Socket.io Connection
# ==========================================
# URL Backend Node.js yang akan menerima data dari ML
BE_URL = os.getenv("BE_URL", "http://localhost:3001")

# ==========================================
# Logging Configuration
# ==========================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")