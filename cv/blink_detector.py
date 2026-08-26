# cv/blink_detector.py
"""
blink_detector.py — Wrapper kompatibilitas ke BlinkEventDetector di eye_fatigue_scoring.py.
"""

from cv.eye_fatigue_scoring import BlinkEventDetector, EyeState

# Alias kelas utama untuk backwards compatibility
BlinkDetector = BlinkEventDetector