# services/aggregator_service.py
#
# Modul ini mengakumulasi data hasil analisis per-frame selama 60 detik,
# lalu menghasilkan satu payload ringkasan (summary) yang dikirim ke Backend
# via Channel B (py-minute-summary).
#
# Data yang diakumulasi per menit:
#   - Durasi "Dekat" dan "Jauh" dalam detik
#   - Total kedipan & rata-rata blink rate
#   - Health status & kondisi mata dari EyeConditionAnalyzer

import time
import threading
from typing import Callable
from utils.time_utils import get_current_iso_time
from utils.logger import get_logger

logger = get_logger(__name__)

# Durasi satu window agregasi (detik)
AGGREGATION_WINDOW_SEC = 60


class AggregatorService:
    """
    Mengakumulasi data deteksi per-frame selama satu menit,
    lalu memanggil callback dengan payload ringkasan untuk dikirim ke BE.
    """

    def __init__(self, on_summary: Callable[[dict], None]):
        """
        Args:
            on_summary: Callback yang dipanggil setiap 1 menit dengan payload summary.
                        Biasanya ini memanggil be_socket_client.emit_minute_summary().
        """
        self._on_summary = on_summary
        self._lock = threading.Lock()
        self._reset_state()

        self._thread: threading.Thread | None = None
        self._running = False

    def _reset_state(self) -> None:
        """Reset semua counter untuk window baru."""
        self._robot_id: str | None = None
        self._period_start: str = get_current_iso_time()
        self._near_sec: float = 0.0
        self._far_sec: float = 0.0
        self._blink_count: int = 0
        self._blink_rate_samples: list[float] = []
        self._health_statuses: list[str] = []
        self._eye_conditions: list[str] = []
        self._recommendations: list[str] = []
        self._last_frame_time: float = time.time()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Mulai background thread yang menjalankan timer 1 menit."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._aggregation_loop,
            daemon=True,
            name="aggregator-service"
        )
        self._thread.start()
        logger.info("[Aggregator] Service dimulai. Window = 60 detik.")

    def stop(self) -> None:
        """Hentikan background thread aggregator."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("[Aggregator] Service dihentikan.")

    def _aggregation_loop(self) -> None:
        """Loop yang menunggu 60 detik lalu memanggil _flush_summary."""
        while self._running:
            time.sleep(AGGREGATION_WINDOW_SEC)
            if self._running:
                self._flush_summary()

    # ------------------------------------------------------------------
    # Data Ingestion (dipanggil tiap frame dari VisionPipelineService)
    # ------------------------------------------------------------------

    def ingest(self, robot_id: str, distance: str, blink_event: bool,
               blink_rate: float, health_status: str,
               eye_conditions: list[str], recommendations: list[str]) -> None:
        """
        Terima data satu frame untuk diakumulasi.

        Args:
            robot_id (str): ID robot.
            distance (str): "Dekat" atau "Jauh".
            blink_event (bool): True jika frame ini mendeteksi kedipan.
            blink_rate (float): Blink rate per menit saat frame ini.
            health_status (str): "Aman" / "Peringatan".
            eye_conditions (list): Kondisi mata terdeteksi.
            recommendations (list): Rekomendasi dari EyeConditionAnalyzer.
        """
        current_time = time.time()

        with self._lock:
            # Simpan robot_id dari frame pertama di window ini
            if self._robot_id is None:
                self._robot_id = robot_id

            # Hitung delta waktu antar frame untuk akurasi durasi
            delta = current_time - self._last_frame_time
            # Batasi delta maksimal 1 detik untuk mencegah jump jika lag
            delta = min(delta, 1.0)
            self._last_frame_time = current_time

            # Akumulasi durasi berdasarkan status jarak
            if distance == "Dekat":
                self._near_sec += delta
            else:
                self._far_sec += delta

            # Akumulasi blink
            if blink_event:
                self._blink_count += 1

            # Simpan sample blink rate
            if blink_rate > 0:
                self._blink_rate_samples.append(blink_rate)

            # Simpan health status dan kondisi
            self._health_statuses.append(health_status)
            self._eye_conditions.extend(eye_conditions)
            self._recommendations.extend(recommendations)

    # ------------------------------------------------------------------
    # Summary Flush
    # ------------------------------------------------------------------

    def _flush_summary(self) -> None:
        """
        Hitung payload ringkasan dari data yang terakumulasi,
        panggil callback, lalu reset state untuk window berikutnya.
        """
        with self._lock:
            robot_id = self._robot_id
            period_start = self._period_start
            near_sec = round(self._near_sec)
            far_sec = round(self._far_sec)
            total_sec = near_sec + far_sec
            blink_count = self._blink_count
            avg_blink_rate = round(
                sum(self._blink_rate_samples) / len(self._blink_rate_samples), 2
            ) if self._blink_rate_samples else 0.0
            dominant_distance = "Dekat" if near_sec >= far_sec else "Jauh"

            # Ambil health_status yang paling sering muncul
            health_status = "Aman"
            if self._health_statuses:
                health_status = max(
                    set(self._health_statuses),
                    key=self._health_statuses.count
                )

            # Deduplicate kondisi dan rekomendasi
            eye_conditions = list(dict.fromkeys(self._eye_conditions))
            recommendations = list(dict.fromkeys(self._recommendations))

            near_percentage = round((near_sec / total_sec) * 100, 1) if total_sec > 0 else 0.0

            period_end = get_current_iso_time()

        if robot_id is None:
            logger.info("[Aggregator] Tidak ada data dalam window ini. Summary dilewati.")
            self._reset_state()
            return

        summary = {
            "robot_id": robot_id,
            "period_start": period_start,
            "period_end": period_end,
            "near_duration_sec": near_sec,
            "far_duration_sec": far_sec,
            "near_percentage": near_percentage,
            "blink_count": blink_count,
            "avg_blink_rate": avg_blink_rate,
            "dominant_distance": dominant_distance,
            "health_status": health_status,
            "eye_conditions": eye_conditions,
            "recommendations": recommendations
        }

        logger.info(
            f"[Aggregator] Summary robot={robot_id} | "
            f"Dekat={near_sec}s | Jauh={far_sec}s | Blink={blink_count}"
        )

        # Panggil callback (emit ke BE)
        try:
            self._on_summary(summary)
        except Exception as e:
            logger.error(f"[Aggregator] Error saat memanggil on_summary callback: {e}")

        # Reset untuk window berikutnya
        self._reset_state()
