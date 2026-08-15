# cv/eye_analyzer.py

class EyeConditionAnalyzer:
    def __init__(self):
        # Mengacu pada standar normal Kaur et al. (2022)
        self.NORMAL_BLINK_MIN = 15
        self.NORMAL_BLINK_MAX = 20
        # Batas kritis penggunaan perangkat digital
        self.CRITICAL_BLINK_RATE = 7 
        # Jarak aman membaca (Widayat et al., 2026) dalam cm
        self.SAFE_DISTANCE_CM = 30 
        # Batas screen time (Ha et al., 2025) dalam jam
        self.MAX_SCREEN_TIME_HOURS = 4 

    def analyze_from_blink_rate(self, blink_rate_per_minute):
        """
        Menganalisis kondisi mata Kering dan Lelah hanya dari frekuensi kedipan.
        """
        conditions = []
        recommendations = []

        if blink_rate_per_minute < self.CRITICAL_BLINK_RATE:
            conditions.append("Mata Lelah (Astenopia) Berat & Mata Kering Kritis")
            recommendations.append("Segera lakukan aturan 20-20-20: Istirahat 20 detik melihat jarak 20 kaki.")
        elif blink_rate_per_minute < self.NORMAL_BLINK_MIN:
            conditions.append("Risiko Mata Kering (Instabilitas Tear Film)")
            recommendations.append("Tingkatkan frekuensi kedipan Anda secara sadar.")
        elif self.NORMAL_BLINK_MIN <= blink_rate_per_minute <= self.NORMAL_BLINK_MAX:
            conditions.append("Normal")
            recommendations.append("Pertahankan frekuensi kedipan Anda.")
        else:
            conditions.append("Normal (Sering Berkedip)")
            
        return {
            "status": "Aman" if "Normal" in conditions[0] else "Peringatan",
            "conditions": conditions,
            "recommendations": recommendations
        }

    def analyze_myopia_risk(self, distance_cm=None, screen_time_hours=None):
        """
        Menganalisis risiko Miopia berdasarkan jarak dan waktu layar.
        """
        risk_level = "Rendah"
        factors = []

        if distance_cm is not None and distance_cm < self.SAFE_DISTANCE_CM:
            risk_level = "Tinggi"
            factors.append(f"Jarak pandang terlalu dekat ({distance_cm} cm).")
            
        if screen_time_hours is not None and screen_time_hours > self.MAX_SCREEN_TIME_HOURS:
            risk_level = "Sangat Tinggi"
            factors.append(f"Screen time berlebih (>4 jam/hari). Risiko 97% lebih tinggi.")

        return {
            "myopia_risk": risk_level,
            "risk_factors": factors
        }