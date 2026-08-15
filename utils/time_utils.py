# utils/time_utils.py

from datetime import datetime, timezone

def get_current_iso_time() -> str:
    """
    Mendapatkan waktu UTC saat ini dan mengembalikannya dalam format string ISO 8601.
    
    Format yang dihasilkan dirancang agar sesuai dengan kontrak API yang diharapkan,
    yaitu 'YYYY-MM-DDTHH:MM:SS.mmmZ' (contoh: 2026-07-19T16:30:15.123Z).

    Returns:
        str: String waktu saat ini dalam format ISO 8601 dengan indikator zona waktu Zulu (Z).
    """
    # Ambil waktu saat ini berdasarkan zona waktu UTC
    now_utc = datetime.now(timezone.utc)
    
    # Format ke ISO 8601 dengan resolusi milidetik
    iso_str = now_utc.isoformat(timespec='milliseconds')
    
    # Python secara default menggunakan '+00:00' untuk UTC, 
    # kita ubah menjadi 'Z' (Zulu) agar sesuai persis dengan kontrak API.
    formatted_iso_str = iso_str.replace('+00:00', 'Z')
    
    return formatted_iso_str