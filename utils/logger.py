# utils/logger.py

import logging
import sys
from config import settings

def get_logger(name: str) -> logging.Logger:
    """
    Membuat dan mengonfigurasi logger terpusat untuk aplikasi.
    
    Fungsi ini membaca LOG_LEVEL dari config/settings.py dan memastikan
    format output log seragam di seluruh modul (timestamp, level, nama modul, pesan).

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

        # Konfigurasi output ke console (stdout)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)

        # Format log: Waktu | LEVEL | Nama Modul | Pesan
        formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

    return logger