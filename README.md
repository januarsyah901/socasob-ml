# SocaSob ML — Sistem Pemantauan Kesehatan Mata

Sistem Computer Vision real-time untuk deteksi mata lelah, mata kering, dan risiko miopia.
Proyek Engineering Physics, semester 5.

---

## Arsitektur

```
Camera (Webcam/ESP32-CAM)
    │
    ▼
MediaPipe Face Mesh (468 landmarks)
    │
    ├──► vision/blink_detector.py ──► vision/metrics_window.py ──► scoring/fatigue_score.py
    │         (EAR, state machine,        (sliding window 60s,         (composite score 0-100,
    │          incomplete blink)           PERCLOS, rate, variability)   hysteresis debounce)
    │
    ├──► vision/distance_estimator.py ──► scoring/active_myopia_guard.py
    │         (pinhole camera model)          (jarak <50cm warning,
    │                                          aturan 20-20-20 timer)
    │
    └──► scoring/myopia_risk.py
              (screen time harian,
               dosis-respons lookup)
    │
    ▼
ml/engine.py — Inference Engine
    ├── FatigueDetector (rule-based → replaceable)
    ├── DryEyeDetector  (rule-based → replaceable)
    └── MyopiaRiskModel (rule-based → replaceable)
    │
    ├──► storage/database.py (SQLite: fatigue_logs, dry_eye_logs, myopia_risk_logs)
    │
    └──► realtime/ws_server.py (WebSocket broadcast + query riwayat)
              ├── Website client (dashboard live + laporan historis)
              └── Hardware client (LCD + speaker, payload ringkas)
```

## Tiga Modul Deteksi

| Modul | Fungsi | Skala Waktu |
|-------|--------|-------------|
| **A** — Fatigue/Dry-Eye | Analisis kedipan (EAR, PERCLOS, durasi, variabilitas) | Real-time, window 60 detik |
| **B1** — Active Guard | Jarak mata-layar + aturan 20-20-20 | Kontinu per detik |
| **B2** — Myopia Risk | Akumulasi screen time harian → kurva dosis-respons | Kumulatif per hari |

## Struktur File

```
camera/
    base.py                 # Interface abstrak CameraSource
    webcam_source.py        # Implementasi webcam (OpenCV)
    esp32_source.py         # Implementasi ESP32-CAM (HTTP MJPEG / WebSocket push)
vision/
    blink_detector.py       # EAR calculation + state machine kedipan + incomplete blink
    distance_estimator.py   # Estimasi jarak mata-layar (pinhole camera model)
    metrics_window.py       # Sliding window 60s: rate, PERCLOS, durasi, variabilitas
scoring/
    fatigue_score.py        # Composite score + tabel klasifikasi + hysteresis
    active_myopia_guard.py  # Modul B1: jarak + timer 20-20-20
    myopia_risk.py          # Modul B2: lookup dosis-respons (Ha dkk., 2025)
ml/
    engine.py               # Inference Engine: 3 detector independen (Protocol-based)
storage/
    database.py             # SQLite: 3 tabel riwayat + query N hari
realtime/
    ws_server.py            # WebSocket server: broadcast + query riwayat
main.py                     # Entry point: wiring semua komponen
```

## Cara Menjalankan

### 1. Setup

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Jalankan (Webcam default)

```bash
python main.py
```

Server WebSocket akan berjalan di `ws://localhost:8765`.

### 3. Jalankan dengan kalibrasi baseline

```bash
CALIBRATE=true python main.py
```

Sistem akan meminta Anda duduk santai 90 detik untuk mengukur blink rate personal.

### 4. Jalankan dengan ESP32-CAM

```bash
VIDEO_SOURCE=esp32 ESP32_STREAM_URL=http://192.168.1.100:81/stream python main.py
```

## Cara Ganti Camera Source

Edit environment variable `VIDEO_SOURCE`:

| Nilai | Sumber |
|-------|--------|
| `webcam` (default) | Webcam lokal USB/internal |
| `esp32` | ESP32-CAM via HTTP MJPEG stream |

Untuk ESP32-CAM, set juga `ESP32_STREAM_URL` ke URL stream Anda.

Untuk menambah sumber baru (misal IP camera), buat class yang inherit `CameraSource`
dari `camera/base.py` dan implementasikan 5 method abstraknya.

## Environment Variables

| Variable | Default | Deskripsi |
|----------|---------|-----------|
| `VIDEO_SOURCE` | `webcam` | Sumber video: `webcam` atau `esp32` |
| `WEBCAM_INDEX` | `0` | Index device webcam |
| `ESP32_STREAM_URL` | — | URL HTTP MJPEG stream ESP32-CAM |
| `EAR_THRESHOLD` | `0.23` | Threshold EAR mata tertutup |
| `WS_HOST` | `0.0.0.0` | Host WebSocket server |
| `WS_PORT` | `8765` | Port WebSocket server |
| `CALIBRATE` | `false` | Aktifkan kalibrasi baseline personal |
| `CALIBRATION_SECONDS` | `90` | Durasi kalibrasi (detik) |
| `DB_PATH` | `data/eye_health.db` | Path file SQLite |
| `DB_LOG_INTERVAL` | `5.0` | Interval logging ke DB (detik) |
| `WS_BROADCAST_INTERVAL` | `1.0` | Interval broadcast WebSocket (detik) |

## Menghubungkan Klien

### Website (JavaScript)

```javascript
const ws = new WebSocket('ws://localhost:8765');

// Terima broadcast real-time
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log(`[${data.type}] Status: ${data.status}`);
};

// Request riwayat
ws.send(JSON.stringify({
    action: 'get_history',
    type: 'fatigue',    // atau 'dry_eye', 'myopia_risk'
    days: 7
}));

// Request ringkasan
ws.send(JSON.stringify({
    action: 'get_summary',
    days: 1
}));
```

### Hardware (ESP32 / Arduino)

```cpp
// Terima broadcast real-time (JSON ringkas)
// Contoh payload:
// {"type":"fatigue","status":"Aman","composite_score":15.2,"timestamp":1234567890}
// {"type":"myopia_risk","distance_warning":true,"break_state":"break_needed"}

// Parsing minimal — cukup baca field "type", "status", dan flag peringatan
// untuk memicu aktuator LCD & speaker.
```

## Arsitektur ML / Inference Engine

Engine menggunakan **Protocol (structural subtyping)** agar detector rule-based
bisa diganti dengan model terlatih tanpa mengubah kontrak data:

```python
# Ganti rule-based fatigue detector dengan model terlatih:
from ml.engine import InferenceEngine

class TrainedFatigueDetector:
    def predict(self, features):
        # Load model, jalankan inference
        return {"type": "fatigue", "status": "...", ...}

engine = InferenceEngine(
    fatigue_detector=TrainedFatigueDetector(),
    # dry_eye dan myopia tetap rule-based
)
```

## Referensi Ilmiah

- **Chai dkk. (2025)**, Scientific Reports, n=45 — ambang composite score
- **Kaur dkk. (2022)**, Ophthalmology and Therapy — aturan 20-20-20, jarak ≥50cm
- **Ha dkk. (2025)**, JAMA Network Open, 45 studi, n=335.524 — dosis-respons myopia risk
- **Dodgson (2004)** — jarak interokular rata-rata 6.3cm

## Stack

Python 3.10+, OpenCV, MediaPipe Face Mesh, NumPy, websockets, SQLite
