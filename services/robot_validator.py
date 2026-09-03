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
        logger.warning(f"[{robot_id}] Validasi Backend gagal (HTTP {e.code}): Robot tidak valid/terdaftar.")
        is_valid = False
    except Exception as e:
        # Backend offline / tidak dapat dihubungi
        logger.warning(f"[{robot_id}] Gagal menghubungi Backend untuk validasi ({e}).")
        ttl = OFFLINE_TTL_SEC
        with _lock:
            if robot_id in _cache:
                is_valid = _cache[robot_id][0]
            else:
                is_valid = False
    finally:
        with _lock:
            _cache[robot_id] = (is_valid, now + ttl)
            _in_flight.discard(robot_id)


def is_robot_registered(robot_id: str) -> bool:
    """
    Mengecek apakah robot_id terdaftar dan berstatus 'active' di Backend.
    Non-blocking: selalu mengembalikan hasil dalam 0ms tanpa pernah menahan thread video.
    Semua robot_id wajib tervalidasi oleh Backend (tidak ada bypass/hardcode).

    Args:
        robot_id (str): ID robot yang divalidasi.

    Returns:
        bool: True jika valid & aktif di database Backend, False jika tidak terdaftar / dinonaktifkan.
    """
    if not robot_id or not isinstance(robot_id, str) or not robot_id.strip():
        return False

    robot_id = robot_id.strip()
    now = time.time()

    with _lock:
        if robot_id in _cache:
            is_valid, expire_time = _cache[robot_id]
            # Jika cache sudah expired dan belum ada background thread berjalan, trigger refresh di background
            if now >= expire_time and robot_id not in _in_flight:
                _in_flight.add(robot_id)
                threading.Thread(target=_validate_in_background, args=(robot_id,), daemon=True, name="robot-val-bg").start()
            return is_valid

        # ID baru yang belum ada di cache: picu validasi ke Backend
        if robot_id not in _in_flight:
            _in_flight.add(robot_id)
            threading.Thread(target=_validate_in_background, args=(robot_id,), daemon=True, name="robot-val-bg").start()
        return False
