# evaluate_dataset.py
"""
Script Evaluasi Kinerja AI Deteksi Kedipan & Kelelahan (SocaSob ML)
------------------------------------------------------------------
Dua Mode Evaluasi REAL (Data Asli):
1. Mode Webcam Live (__webcam): Merekam & mengevaluasi otomatis sinyal kedipan via webcam selama N detik (default: 300 detik / 5 menit).
2. Mode File Video (__video): Evaluasi file video .mp4/.avi yang disiapkan.

Menghasilkan file grafik real: `eval_real_result_chart.png` (Data Asli AI 5 Menit).
"""

import argparse
import sys
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

from cv.face_mesh import FaceMeshDetector
from vision.blink_detector import (
    BlinkEventDetector,
    calculate_ear,
    extract_eye_coordinates,
)
from vision.metrics_window import MetricsWindow
from ml.engine import InferenceEngine
from scoring.active_myopia_guard import ActiveMyopiaGuard
from scoring.myopia_risk import MyopiaRiskEstimator
from config import settings


def run_evaluation(source_type: str = "webcam", video_path: str = None, duration_seconds: int = 300, ear_threshold: float = 0.22):
    minutes = duration_seconds / 60.0
    print("=" * 65)
    print(f"  SOCACOMVI — EVALUASI KINERJA AI DATA ASLI ({minutes:.1f} MENIT)")
    print("=" * 65)

    if source_type == "webcam":
        print(f"[INFO] Mengambil & menganalisis data sensor webcam selama {duration_seconds} detik ({minutes:.1f} menit)...")
        print("Silakan beraktivitas di depan layar/webcam seperti biasa...\n")
        cap = cv2.VideoCapture(0)
    else:
        print(f"[INFO] Memproses File Video: {video_path}")
        cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[ERROR] Gagal membuka sumber video: {video_path or 'Webcam (Device 0)'}")
        return

    face_mesh = FaceMeshDetector()
    blink_detector = BlinkEventDetector(ear_threshold=ear_threshold)
    metrics_window = MetricsWindow(window_seconds=60)
    myopia_guard = ActiveMyopiaGuard()
    myopia_risk = MyopiaRiskEstimator()
    engine = InferenceEngine()

    ear_history = []
    closed_history = []
    blink_timestamps = []
    perclos_history = []
    blink_rate_history = []
    fatigue_score_history = []
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0 or source_type == "webcam":
        fps = 30.0

    frame_idx = 0
    start_time = time.time()
    detected_blinks = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        now = time.time()
        elapsed = now - start_time
        h, w = frame.shape[:2]

        landmarks = face_mesh.process(frame)
        face_confidence = 1.0 if landmarks is not None else 0.0
        avg_ear = 0.0

        if landmarks is not None:
            left_eye, right_eye = extract_eye_coordinates(landmarks, w, h)
            ear_l = calculate_ear(left_eye)
            ear_r = calculate_ear(right_eye)
            avg_ear = (ear_l + ear_r) / 2.0

        is_closed = (avg_ear < ear_threshold) if landmarks is not None else False
        ear_history.append(avg_ear)
        closed_history.append(1 if is_closed else 0)

        # Feed MetricsWindow with actual wall-clock timestamp now
        metrics_window.add_frame(now, is_closed=is_closed, is_valid=(face_confidence >= 0.5))
        blink_event = blink_detector.update(avg_ear, face_confidence, now)
        if blink_event:
            detected_blinks += 1
            blink_timestamps.append(elapsed)
            # Convert BlinkEvent object to dictionary if needed
            event_dict = {
                "timestamp": getattr(blink_event, "timestamp", now),
                "duration": getattr(blink_event, "duration", 0.2),
                "incomplete": getattr(blink_event, "incomplete", False)
            }
            metrics_window.add_blink(event_dict)

        # Run Engine to calculate real PERCLOS & composite fatigue score
        guard_res = myopia_guard.update(landmarks is not None, None, now)
        risk_res = myopia_risk.get_risk()
        engine_res = engine.run(metrics_window, guard_res, risk_res)

        fatigue_data = engine_res.get("fatigue", {})
        dry_eye_data = engine_res.get("dry_eye", {})

        current_perclos = metrics_window.perclos() * 100.0
        current_rate = metrics_window.smoothed_blink_rate()
        current_score = fatigue_data.get("composite_score", 0.0)

        perclos_history.append(current_perclos)
        blink_rate_history.append(current_rate)
        fatigue_score_history.append(current_score)

        # Terminal Progress Bar
        if source_type == "webcam":
            progress = min(1.0, elapsed / duration_seconds)
            bar_len = 30
            filled_len = int(bar_len * progress)
            bar = '=' * filled_len + '-' * (bar_len - filled_len)
            mins_left = max(0, (duration_seconds - elapsed) / 60.0)
            sys.stdout.write(f"\r[PENGUJIAN 5 MENIT] [{bar}] {progress*100:.1f}% | EAR: {avg_ear:.3f} | Kedipan: {detected_blinks} | Skor Kelelahan: {current_score:.1f} | Sisa: {mins_left:.1f}m")
            sys.stdout.flush()

            if elapsed >= duration_seconds:
                break

    cap.release()
    cv2.destroyAllWindows()
    print("\n")

    duration = elapsed if source_type == "webcam" else (frame_idx / fps)
    total_mins = duration / 60.0
    print("=" * 65)
    print(f"        HASIL EVALUASI REAL {total_mins:.1f} MENIT (DATA ASLI SENSOR AI)")
    print("=" * 65)
    print(f" Total Durasi Uji        : {duration:.1f} detik / {total_mins:.2f} menit ({frame_idx} frame)")
    print(f" Total Kedipan Terdeteksi: {detected_blinks} kali")
    if duration > 0:
        print(f" Rata-rata Blink Rate   : {(detected_blinks / total_mins):.1f} kedipan/menit")
    if ear_history:
        print(f" Rata-rata EAR Mata     : {np.mean(ear_history):.3f}")
        print(f" Nilai EAR Minimum      : {np.min(ear_history):.3f}")
        print(f" Nilai EAR Maksimum     : {np.max(ear_history):.3f}")
        print(f" Rata-rata PERCLOS (%)  : {np.mean(perclos_history):.2f}%")
        print(f" Rata-rata Skor Kelelahan: {np.mean(fatigue_score_history):.1f} / 100")
    print("=" * 65)

    # Multi-panel 5-Minute Evaluation Chart
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    time_axis = np.linspace(0, duration / 60.0, len(ear_history))

    # Panel 1: EAR Waveform & Blink Markers
    ax1.plot(time_axis, ear_history, color='#1f77b4', linewidth=1.2, label='Sinyal EAR Live')
    ax1.axhline(y=ear_threshold, color='#ff7f0e', linestyle='--', linewidth=1.5, label=f'Threshold Kedipan ({ear_threshold})')
    for ts in blink_timestamps:
        ax1.axvline(x=ts / 60.0, color='#2ca02c', alpha=0.35, linestyle=':', linewidth=1.0)
    ax1.set_ylabel('EAR (Eye Aspect Ratio)', fontsize=10, fontweight='bold')
    ax1.set_title(f'1. Sinyal Live EAR Mata & Kedipan ({detected_blinks} Kedipan)', fontsize=11, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # Panel 2: PERCLOS & Blink Rate Trend
    ax2.plot(time_axis, perclos_history, color='#9467bd', linewidth=1.8, label='PERCLOS (% Mata Terpejam)')
    ax2_rate = ax2.twinx()
    ax2_rate.plot(time_axis, blink_rate_history, color='#17becf', linestyle='-.', linewidth=1.5, label='Blink Rate (kedipan/min)')
    ax2.set_ylabel('PERCLOS (%)', color='#9467bd', fontsize=10, fontweight='bold')
    ax2_rate.set_ylabel('Blink Rate (/min)', color='#17becf', fontsize=10, fontweight='bold')
    ax2.set_title('2. Tren PERCLOS (Mata Terpejam) & Blink Rate (Sliding Window 60s)', fontsize=11, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)

    # Panel 3: Composite Fatigue Score
    ax3.plot(time_axis, fatigue_score_history, color='#d62728', linewidth=2.0, label='Composite Fatigue Score (0-100)')
    ax3.axhline(y=30, color='#e67e22', linestyle='--', label='Batas Peringatan Ringan (30)')
    ax3.axhline(y=60, color='#c0392b', linestyle='--', label='Batas Peringatan Berat (60)')
    ax3.set_ylabel('Fatigue Score (0-100)', fontsize=10, fontweight='bold')
    ax3.set_xlabel('Waktu (menit)', fontsize=11, fontweight='bold')
    ax3.set_ylim(-5, 105)
    ax3.set_title('3. Composite Fatigue Score (0-100) & Hysteresis Classification', fontsize=11, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    output_img = "eval_real_result_chart.png"
    plt.savefig(output_img, dpi=300)
    plt.close()

    print(f"\n[SUKSES] Grafik evaluasi 5 MENIT ASLI berhasil disimpan ke: {output_img}\n")


def generate_sample_confusion_matrix_plot():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    cm = np.array([[850, 40],
                   [ 25, 185]])

    im = ax1.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax1.figure.colorbar(im, ax=ax1)
    classes = ['Eye Open', 'Eye Closed']
    ax1.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title='Confusion Matrix Deteksi Mata (Sintetis/Contoh)',
           ylabel='True Label',
           xlabel='Predicted Label')

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax1.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontweight='bold', fontsize=12)

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    scores = [94.1, 82.2, 88.1, 85.0]
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#f1c40f']

    bars = ax2.bar(metrics, scores, color=colors, width=0.5)
    ax2.set_ylabel('Persentase (%)', fontsize=11)
    ax2.set_title('Contoh Laporan Metrik AI', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 110)
    for bar in bars:
        h = bar.get_height()
        ax2.annotate(f'{h:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    output_img = "ai_performance_evaluation.png"
    plt.savefig(output_img, dpi=300)
    plt.close()
    return output_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluasi Performa AI Deteksi Kedipan & Kelelahan")
    parser.add_argument("--webcam", action="store_true", help="Uji langsung via webcam real-time")
    parser.add_argument("--duration", type=int, default=300, help="Durasi pengujian webcam dalam detik (default: 300 = 5 menit)")
    parser.add_argument("--video", type=str, help="Path ke file video uji (.mp4 / .avi)")
    parser.add_argument("--ear_thresh", type=float, default=0.22, help="Threshold EAR (default: 0.22)")
    parser.add_argument("--demo_chart", action="store_true", help="Buat grafik contoh Confusion Matrix")

    args = parser.parse_args()

    if args.webcam:
        run_evaluation(source_type="webcam", duration_seconds=args.duration, ear_threshold=args.ear_thresh)
    elif args.video:
        run_evaluation(source_type="video", video_path=args.video, ear_threshold=args.ear_thresh)
    else:
        print("[INFO] Menghasilkan grafik contoh (sintetis)...")
        img_path = generate_sample_confusion_matrix_plot()
        print(f"[SUKSES] Grafik sampel tersimpan di: {img_path}")
        print("\nIngin membuat grafik ASLI 5 Menit dari data nyata Anda?")
        print("1. Tes via Webcam Live  : python evaluate_dataset.py --webcam --duration 300")
        print("2. Tes via File Video   : python evaluate_dataset.py --video path/to/video.mp4")
