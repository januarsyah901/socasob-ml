# services/robot_validator.py
#
# Memvalidasi apakah robot_id terdaftar dan aktif di Backend (BE).
# Menggunakan fully asynchronous background validation agar latensi per-frame selalu 0ms murni.

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
_in_flight: set[str] = set()

CACHE_TTL_SEC = 30.0    # 30 detik jika sukses
OFFLINE_TTL_SEC = 60.0  # 60 detik jika BE offline


def _validate_in_background(robot_id: str) -> None:
    """Melakukan request HTTP ke Backend di background thread agar tidak membekukan stream."""
    now = time.time()
    url = f"{settings.BE_URL.rstrip('/')}/api/robots/validate/{robot_id}"
    is_valid = False
    ttl = CACHE_TTL_SEC

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "SocaSob-ML-Validator/1.0"})
        with urllib.request.urlopen(req, timeout=2.0) as response:
            if response.getcode() == 200:
                data = json.loads(response.read().decode('utf-8'))
                is_valid = bool(data.get("valid", False))
            else:
                is_valid = False
    except urllib.error.HTTPError as e:
        is_valid = False
    except Exception as e:
        # Backend offline / tidak dapat dihubungi
        ttl = OFFLINE_TTL_SEC
        with _lock:
            if robot_id in _cache:
                is_valid = _cache[robot_id][0]
            else:
                is_valid = (robot_id == "fadfa566")
    finally:
        with _lock:
            _cache[robot_id] = (is_valid, now + ttl)
            _in_flight.discard(robot_id)


def is_robot_registered(robot_id: str) -> bool:
    """
    Mengecek apakah robot_id terdaftar dan berstatus 'active' di Backend.
    Non-blocking: selalu mengembalikan hasil dalam 0ms tanpa pernah menahan thread video.

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
            # Jika cache sudah expired dan belum ada background thread berjalan, trigger refresh di background
            if now >= expire_time and robot_id not in _in_flight:
                _in_flight.add(robot_id)
                threading.Thread(target=_validate_in_background, args=(robot_id,), daemon=True, name="robot-val-bg").start()
            return is_valid

        # Jika robot bawaan/default belum di-cache, izinkan instan dan validasi di background
        if robot_id == "fadfa566":
            _cache[robot_id] = (True, now + CACHE_TTL_SEC)
            if robot_id not in _in_flight:
                _in_flight.add(robot_id)
                threading.Thread(target=_validate_in_background, args=(robot_id,), daemon=True, name="robot-val-bg").start()
            return True

        # ID baru yang belum dikenal: mulai validasi di background
        if robot_id not in _in_flight:
            _in_flight.add(robot_id)
            threading.Thread(target=_validate_in_background, args=(robot_id,), daemon=True, name="robot-val-bg").start()
        return False
