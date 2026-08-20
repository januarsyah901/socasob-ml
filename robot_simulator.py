#!/usr/bin/env python3
"""
robot_simulator.py — Simulator Robot ESP32-CAM untuk Testing

Script ini mensimulasikan Robot yang:
  1. Connect ke ML Server via WebSocket
  2. Mengirim frame dummy (gambar hitam/kotak sederhana) + distance_json tiap N ms
  3. Mensimulasikan perubahan jarak secara otomatis (Dekat/Jauh bergantian)

Cara pakai:
  python3 robot_simulator.py
  python3 robot_simulator.py --ml-url ws://localhost:5000 --robot-id myrobot123 --fps 10

Requirement:
  pip install websocket-client opencv-python-headless numpy
"""

import argparse
import json
import time
import threading
import random
import numpy as np
import cv2
import websocket

# ─────────────────────────────────────────────
# Konfigurasi Default
# ─────────────────────────────────────────────
DEFAULT_ML_URL   = "http://localhost:5000"
DEFAULT_ROBOT_ID = "robot-simulator-001"
DEFAULT_FPS      = 10       # frame per detik yang dikirim ke ML
DEFAULT_DURATION = 120      # detik total simulasi (0 = selamanya)

def build_dummy_frame(width=320, height=240, distance: str = "Jauh") -> bytes:
    """
    Buat frame dummy (gambar BGR sederhana) lalu encode ke JPEG bytes.
    Warna berubah sesuai status jarak agar mudah dibedakan secara visual.
    """
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    if distance == "Dekat":
        # Latar merah muda → simulasi wajah dekat
        frame[:] = (60, 60, 200)
        label = "DEKAT"
        color = (0, 0, 255)
    else:
        # Latar biru → simulasi wajah jauh
        frame[:] = (200, 100, 60)
        label = "JAUH"
        color = (255, 200, 0)

    # Gambar "wajah" sederhana (lingkaran)
    cx, cy = width // 2, height // 2
    cv2.circle(frame, (cx, cy - 20), 50, (200, 200, 200), -1)          # kepala
    cv2.circle(frame, (cx - 18, cy - 25), 8, (50, 50, 50), -1)         # mata kiri
    cv2.circle(frame, (cx + 18, cy - 25), 8, (50, 50, 50), -1)         # mata kanan
    cv2.ellipse(frame, (cx, cy), (20, 10), 0, 0, 180, (50, 50, 50), 2) # mulut

    # Label jarak
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Encode ke JPEG
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return buf.tobytes()


class RobotSimulator:
    def __init__(self, ml_url: str, robot_id: str, fps: int, duration: int):
        self.ml_url   = ml_url
        self.robot_id = robot_id
        self.fps      = fps
        self.duration = duration
        self.interval = 1.0 / fps

        self.sio = None
        self.running = False
        self.frame_count = 0
        self.start_time = None

        # State jarak — berganti tiap 10 detik
        self.distance = "Jauh"
        self.confidence = 92
        self._distance_toggle_interval = 10  # detik
        self._last_toggle = time.time()

    def _toggle_distance(self):
        """Bergantian antara Dekat dan Jauh setiap N detik."""
        now = time.time()
        if now - self._last_toggle >= self._distance_toggle_interval:
            self.distance = "Dekat" if self.distance == "Jauh" else "Jauh"
            self.confidence = random.randint(88, 98)
            self._last_toggle = now
            print(f"  [Sim] ↕ Jarak berubah → {self.distance} (confidence={self.confidence})")

    def _send_loop(self, sio):
        """Loop utama pengiriman frame ke ML via Socket.io."""
        print(f"\n[Sim] Mulai kirim frame ke ML @ {self.fps} fps...")
        print(f"[Sim] Robot ID : {self.robot_id}")
        print(f"[Sim] ML URL   : {self.ml_url}")
        print(f"[Sim] Durasi   : {'∞' if self.duration == 0 else f'{self.duration} detik'}")
        print("-" * 50)

        self.start_time = time.time()

        while self.running:
            loop_start = time.time()

            # Cek durasi
            if self.duration > 0 and (time.time() - self.start_time) >= self.duration:
                print(f"\n[Sim] Durasi {self.duration} detik selesai. Simulator berhenti.")
                self.running = False
                break

            # Toggle jarak otomatis
            self._toggle_distance()

            # Buat frame & payload
            frame_bytes = build_dummy_frame(distance=self.distance)
            payload = {
                "robot_id": self.robot_id,
                "frame": frame_bytes,
                "distance_json": {
                    "distance": self.distance,
                    "confidence": self.confidence
                }
            }

            try:
                sio.emit("robot-frame", payload)
                self.frame_count += 1

                elapsed = time.time() - self.start_time
                if self.frame_count % (self.fps * 5) == 0:  # log tiap 5 detik
                    print(f"  [Sim] Frame #{self.frame_count:04d} | t={elapsed:.1f}s | {self.distance} ({self.confidence}%)")

            except Exception as e:
                print(f"  [Sim] ⚠ Gagal kirim frame: {e}")

            # Jaga interval
            elapsed_loop = time.time() - loop_start
            sleep_time = self.interval - elapsed_loop
            if sleep_time > 0:
                time.sleep(sleep_time)

    def run(self):
        """Connect ke ML Server dan mulai simulasi."""
        import socketio as sio_lib

        sio = sio_lib.Client()
        self.running = True

        @sio.event
        def connect():
            print(f"[Sim] ✅ Terhubung ke ML Server ({self.ml_url})")
            send_thread = threading.Thread(target=self._send_loop, args=(sio,), daemon=True)
            send_thread.start()

        @sio.event
        def disconnect():
            print("[Sim] ❌ Terputus dari ML Server.")
            self.running = False

        @sio.event
        def connect_error(data):
            print(f"[Sim] ❌ Gagal connect: {data}")
            self.running = False

        print(f"[Sim] Menghubungkan ke ML Server di {self.ml_url} ...")
        try:
            sio.connect(self.ml_url, transports=["websocket"])
            sio.wait()
        except KeyboardInterrupt:
            print("\n[Sim] Dihentikan oleh user (Ctrl+C).")
        except Exception as e:
            print(f"[Sim] Error: {e}")
        finally:
            self.running = False
            if sio.connected:
                sio.disconnect()
            print(f"\n[Sim] Total frame terkirim: {self.frame_count}")


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot Simulator untuk SocaSob")
    parser.add_argument("--ml-url",   default=DEFAULT_ML_URL,   help="URL ML server (default: http://localhost:5000)")
    parser.add_argument("--robot-id", default=DEFAULT_ROBOT_ID, help="ID unik robot (default: robot-simulator-001)")
    parser.add_argument("--fps",      type=int, default=DEFAULT_FPS,      help="FPS pengiriman frame (default: 10)")
    parser.add_argument("--duration", type=int, default=DEFAULT_DURATION, help="Durasi simulasi detik, 0=∞ (default: 120)")
    args = parser.parse_args()

    simulator = RobotSimulator(
        ml_url=args.ml_url,
        robot_id=args.robot_id,
        fps=args.fps,
        duration=args.duration
    )
    simulator.run()
