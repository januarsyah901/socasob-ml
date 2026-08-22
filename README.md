# SocaSob ML

Vision pipeline SocaSob — Flask + MediaPipe. Deteksi kedipan, EAR, stream MJPEG. Deploy via CapRover.

---

### 🌐 Live Production Deployment
- **URL Publik ML Service**: [socasob-ml.hallojanu.xyz](https://socasob-ml.hallojanu.xyz) (Status: 🟡 Active - gthread/eventlet initializing)
- **Frontend App**: [socasob.hallojanu.xyz](https://socasob.hallojanu.xyz)
- **Backend API**: [be-socasob.hallojanu.xyz](https://be-socasob.hallojanu.xyz)

---

## Endpoints

| Method | Path | Deskripsi |
|--------|------|-----------|
| GET | `/` | Dashboard HTML |
| GET | `/video_feed` | Stream MJPEG |
| GET | `/api/features` | JSON fitur mata (kontrak utama) |
| GET | `/health` | Health check CapRover |

## Local

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python app.py
```

Server: `http://localhost:5000`

## Docker

```bash
docker build -t socasob-ml .
docker run --rm -p 5000:5000 socasob-ml
```

Webcam di container butuh device passthrough (`--device /dev/video0`). Di CapRover biasanya `VIDEO_SOURCE=esp32` + `ESP32_STREAM_URL`.

## CapRover

1. App baru, port **5000**
2. Method: Dockerfile (`captain-definition` sudah ada)
3. Health check: `/health`
4. Env (optional):

| Variabel | Default | Ket |
|----------|---------|-----|
| `PORT` | 5000 | CapRover inject ini |
| `VIDEO_SOURCE` | webcam | `webcam` atau `esp32` |
| `WEBCAM_INDEX` | 0 | index kamera lokal |
| `ESP32_STREAM_URL` | — | URL stream ESP32-CAM |
| `EAR_THRESHOLD` | 0.20 | threshold mata tertutup |
| `CONSEC_FRAMES` | 3 | frame min untuk blink |
| `LOG_LEVEL` | INFO | |

Gunicorn: 1 worker, 8 threads. Camera/MediaPipe state in-memory — jangan scale workers.

## Stack

Flask, OpenCV headless, MediaPipe, Gunicorn
