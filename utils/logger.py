# utils/logger.py

import logging
import sys
import collections
import threading
import queue
from config import settings


# ==========================================
# In-Process Log Stream Buffer
# Menyimpan log terbaru & mendistribusikan ke SSE subscribers
# ==========================================

class LogStreamBroadcaster:
    """
    Singleton broadcaster yang menampung log terbaru dari semua logger
    dan mendistribusikannya ke SSE client yang subscribe via /api/logs.

    Cara kerja:
    - Setiap log baru ditaruh ke `_buffer` (circular, maks 500 entri)
    - Setiap SSE client mendapat queue sendiri; broadcaster push ke semua queue aktif
    """
    _instance = None
    _lock = threading.Lock()

    MAX_BUFFER = 500

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._buffer = collections.deque(maxlen=cls.MAX_BUFFER)
                cls._instance._subscribers: list[queue.Queue] = []
                cls._instance._sub_lock = threading.Lock()
        return cls._instance

    def emit(self, formatted_line: str):
        """Simpan ke buffer dan push ke semua subscriber yang aktif."""
        self._buffer.append(formatted_line)
        dead = []
        with self._sub_lock:
            for q in self._subscribers:
                try:
                    q.put_nowait(formatted_line)
                except queue.Full:
                    dead.append(q)
            for q in dead:
                self._subscribers.remove(q)

    def subscribe(self) -> queue.Queue:
        """Buat subscriber baru. Langsung isi dengan buffer historis."""
        q = queue.Queue(maxsize=200)
        # Kirim riwayat log (non-blocking)
        for line in list(self._buffer):
            try:
                q.put_nowait(line)
            except queue.Full:
                break
        with self._sub_lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue):
        """Hapus subscriber saat koneksi SSE ditutup."""
        with self._sub_lock:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass

    @property
    def buffer_snapshot(self) -> list[str]:
        return list(self._buffer)


# Singleton instance global
log_broadcaster = LogStreamBroadcaster()


class BroadcastHandler(logging.Handler):
    """
    Logging handler yang forward semua log ke LogStreamBroadcaster
    agar bisa di-stream ke browser via SSE.
    """
    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
            log_broadcaster.emit(msg)
        except Exception:
            pass  # Jangan sampai log handler crash aplikasi


def get_logger(name: str) -> logging.Logger:
    """
    Membuat dan mengonfigurasi logger terpusat untuk aplikasi.
    
    Fungsi ini membaca LOG_LEVEL dari config/settings.py dan memastikan
    format output log seragam di seluruh modul (timestamp, level, nama modul, pesan).
    Selain ke console, semua log juga dikirim ke LogStreamBroadcaster agar
    bisa ditampilkan realtime di dashboard via SSE (/api/logs).

    Args:
        name (str): Nama dari logger (biasanya menggunakan __name__ dari modul pemanggil).

    Returns:
        logging.Logger: Instance logger yang sudah dikonfigurasi.
    """
    logger = logging.getLogger(name)
    
    # Mencegah penambahan handler ganda jika logger sudah dikonfigurasi sebelumnya
    if not logger.handlers:
        # Mengambil level log dari settings
        log_level_str = getattr(settings, 'LOG_LEVEL', 'INFO').upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        logger.setLevel(log_level)

        # Format log: Waktu | LEVEL | Nama Modul | Pesan
        formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Konfigurasi output ke console (stdout)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Handler kedua: push ke LogStreamBroadcaster → SSE dashboard
        broadcast_handler = BroadcastHandler()
        broadcast_handler.setLevel(log_level)
        broadcast_handler.setFormatter(formatter)
        logger.addHandler(broadcast_handler)

    return logger