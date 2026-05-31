# ============================================================
# tests/test_classifier.py - Unit tests for posture classifier
# ============================================================

import sys
sys.path.insert(0, "..")

from posture_classifier import calculate_angle, classify_posture


class MockLandmark:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z


def make_landmarks(hip_angle_type):
    """
    Create 33 mock landmarks simulating different postures.
    We only need to set the key joints accurately.
    """
    lm = [MockLandmark(0.5, 0.5) for _ in range(33)]

    if hip_angle_type == "QIYAM":
        # Upright: shoulder above hip above knee
        lm[11] = MockLandmark(0.4, 0.2)   # L_SHOULDER
        lm[12] = MockLandmark(0.6, 0.2)
        lm[23] = MockLandmark(0.4, 0.5)   # L_HIP
        lm[24] = MockLandmark(0.6, 0.5)
        lm[25] = MockLandmark(0.4, 0.8)   # L_KNEE
        lm[26] = MockLandmark(0.6, 0.8)
        lm[27] = MockLandmark(0.4, 0.95)  # L_ANKLE
        lm[28] = MockLandmark(0.6, 0.95)
        lm[0]  = MockLandmark(0.5, 0.1)   # NOSE

    elif hip_angle_type == "SUJUD":
        # Prostration: shoulders at or below hip level
        lm[11] = MockLandmark(0.4, 0.75)  # L_SHOULDER (low)
        lm[12] = MockLandmark(0.6, 0.75)
        lm[23] = MockLandmark(0.4, 0.70)  # L_HIP
        lm[24] = MockLandmark(0.6, 0.70)
        lm[25] = MockLandmark(0.4, 0.85)
        lm[26] = MockLandmark(0.6, 0.85)
        lm[27] = MockLandmark(0.4, 0.95)
        lm[28] = MockLandmark(0.6, 0.95)
        lm[0]  = MockLandmark(0.5, 0.80)  # NOSE very low

    return lm


class TestCalculateAngle:

    def test_straight_line_180(self):
        a = MockLandmark(0, 1)
        b = MockLandmark(0, 0)
        c = MockLandmark(0, -1)
        angle = calculate_angle(a, b, c)
        assert abs(angle - 180.0) < 1.0

    def test_right_angle_90(self):
        a = MockLandmark(0, 1)
        b = MockLandmark(0, 0)
        c = MockLandmark(1, 0)
        angle = calculate_angle(a, b, c)
        assert abs(angle - 90.0) < 1.0

    def test_angle_always_positive(self):
        a = MockLandmark(1, 0)
        b = MockLandmark(0, 0)
        c = MockLandmark(0, 1)
        angle = calculate_angle(a, b, c)
        assert angle >= 0

    def test_angle_never_exceeds_180(self):
        a = MockLandmark(1, 0)
        b = MockLandmark(0, 0)
        c = MockLandmark(-1, -0.1)
        angle = calculate_angle(a, b, c)
        assert angle <= 180.0


class TestClassifier:

    def test_qiyam_detected(self):
        lm = make_landmarks("QIYAM")
        result = classify_posture(lm)
        assert result == "QIYAM"

    def test_sujud_detected(self):
        lm = make_landmarks("SUJUD")
        result = classify_posture(lm)
        assert result == "SUJUD"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])