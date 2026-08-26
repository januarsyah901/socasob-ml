# cv/blink_counter.py
"""
blink_counter.py — Wrapper kompatibilitas ke MetricsWindow di eye_fatigue_scoring.py.
"""

from cv.eye_fatigue_scoring import MetricsWindow

# Alias kelas utama untuk backwards compatibility
BlinkCounter = MetricsWindow