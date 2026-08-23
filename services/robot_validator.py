# services/robot_validator.py
#
# Memvalidasi apakah robot_id terdaftar dan aktif di Backend (BE).
# Menggunakan in-memory cache dengan TTL agar performa validasi per-frame tetap 0ms.

import time
import urllib.request
import json
import threading

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# Cache: robot_id -> (is_valid: bool, expire_time: float)
_cache: dict[str, tuple[bool, float]] = {}
_lock = threading.Lock()
CACHE_TTL_SEC = 15.0  # 15 detik


def is_robot_registered(robot_id: str) -> bool:
    """
    Mengecek apakah robot_id terdaftar dan berstatus 'active' di Backend.
    Hasil di-cache selama 15 detik untuk efisiensi tinggi.

    Args:
        robot_id (str): ID robot yang divalidasi.

    Returns:
        bool: True jika valid & aktif, False jika tidak terdaftar / dinonaktifkan.
    """
    if not robot_id:
        return False

    now = time.time()

    with _lock:
        if robot_id in _cache:
            is_valid, expire_time = _cache[robot_id]
            if now < expire_time:
                return is_valid

    # Fetch ke Backend: /api/robots/validate/{robot_id}
    url = f"{settings.BE_URL.rstrip('/')}/api/robots/validate/{robot_id}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "SocaSob-ML-Validator/1.0"})
        with urllib.request.urlopen(req, timeout=3.0) as response:
            if response.getcode() == 200:
                data = json.loads(response.read().decode('utf-8'))
                is_valid = bool(data.get("valid", False))
            else:
                is_valid = False
    except urllib.error.HTTPError as e:
        if e.code == 404:
            is_valid = False
        else:
            logger.warning(f"[RobotValidator] HTTP error {e.code} saat validasi robot {robot_id}")
            # Jika server BE mengembalikan error 5xx, izinkan sementara jika ada di cache
            is_valid = False
    except Exception as e:
        logger.warning(f"[RobotValidator] Gagal menghubungi BE untuk validasi robot {robot_id}: {e}")
        # Jika BE tidak dapat dihubungi, jangan matikan robot yang sebelumnya aktif jika dalam toleransi
        with _lock:
            if robot_id in _cache:
                return _cache[robot_id][0]
        # Default allow robot bawaan 'fadfa566' jika offline, tolak ID asing
        is_valid = (robot_id == "fadfa566")

    with _lock:
        _cache[robot_id] = (is_valid, now + CACHE_TTL_SEC)

    return is_valid
