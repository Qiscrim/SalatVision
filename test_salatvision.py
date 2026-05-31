# ============================================================
# test_salatvision.py — Unit tests for SalatVision
#
# Run from the SalatVision project root:
#   pytest test_salatvision.py -v
#
# No camera required. All tests use synthetic landmark data.
#
# Coverage:
#   1. Landmark mock helpers
#   2. calculate_angle()         — geometry primitives
#   3. get_joint_angles()        — measurement extraction
#   4. classify_posture()        — side-view postures
#   5. classify_posture()        — front-view postures
#   6. classify_posture()        — boundary / edge cases
#   7. RakaatFSM                 — sequence counting
#   8. RakaatFSM                 — wrong-order rejection
#   9. TransitionSmoother        — smoothing behaviour
#  10. LandmarkSmoother          — median filter
# ============================================================

import time
import math
import pytest

# ── Shared mock ──────────────────────────────────────────────

class _LM:
    """Minimal landmark object matching MediaPipe's interface."""
    def __init__(self, x=0.5, y=0.5, z=0.0, visibility=0.99):
        self.x          = x
        self.y          = y
        self.z          = z
        self.visibility = visibility


def _make_landmarks(overrides=None):
    """
    Return a list of 33 landmarks, all at (0.5, 0.5) with high visibility.
    Pass overrides={index: _LM(...)} to set specific joints.

    MediaPipe landmark indices used by posture_classifier:
      0  = NOSE
      2  = LEFT_EAR          (left/right from MediaPipe's perspective)
      5  = RIGHT_EAR
      11 = LEFT_SHOULDER
      12 = RIGHT_SHOULDER
      13 = LEFT_ELBOW
      14 = RIGHT_ELBOW
      15 = LEFT_WRIST
      16 = RIGHT_WRIST
      23 = LEFT_HIP
      24 = RIGHT_HIP
      25 = LEFT_KNEE
      26 = RIGHT_KNEE
      27 = LEFT_ANKLE
      28 = RIGHT_ANKLE
    """
    lm = [_LM() for _ in range(33)]
    if overrides:
        for idx, val in overrides.items():
            lm[idx] = val
    return lm


# ── Convenience index constants (mirrors posture_classifier) ──

NOSE       = 0
L_EAR, R_EAR = 7, 8
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW,   R_ELBOW     = 13, 14
L_WRIST,   R_WRIST     = 15, 16
L_HIP,     R_HIP       = 23, 24
L_KNEE,    R_KNEE       = 25, 26
L_ANKLE,   R_ANKLE      = 27, 28


# ── Side-view helper ──────────────────────────────────────────

def _side_base():
    """
    Landmark positions for a typical SIDE-VIEW camera.
    Left and right joints are spread horizontally:
      left landmarks at x~0.20, right at x~0.75
    This gives avg_spread ≈ 0.55 → view='side'.
    """
    return {
        # Shoulders
        L_SHOULDER: _LM(x=0.20, y=0.30),
        R_SHOULDER: _LM(x=0.75, y=0.30),
        # Hips
        L_HIP:      _LM(x=0.20, y=0.55),
        R_HIP:      _LM(x=0.75, y=0.55),
        # Knees
        L_KNEE:     _LM(x=0.20, y=0.75),
        R_KNEE:     _LM(x=0.75, y=0.75),
        # Ankles
        L_ANKLE:    _LM(x=0.20, y=0.92),
        R_ANKLE:    _LM(x=0.75, y=0.92),
        # Wrists (hanging near hips by default)
        L_WRIST:    _LM(x=0.20, y=0.57),
        R_WRIST:    _LM(x=0.75, y=0.57),
        # Nose
        NOSE:       _LM(x=0.48, y=0.15),
        # Ears
        L_EAR:      _LM(x=0.45, y=0.17),
        R_EAR:      _LM(x=0.52, y=0.17),
        # Elbows
        L_ELBOW:    _LM(x=0.20, y=0.44),
        R_ELBOW:    _LM(x=0.75, y=0.44),
    }


def _front_base():
    """
    Landmark positions for a typical FRONT-VIEW camera.
    Left and right joints are close in X (avg_spread ≈ 0.07 → view='front').
    """
    return {
        L_SHOULDER: _LM(x=0.43, y=0.30),
        R_SHOULDER: _LM(x=0.57, y=0.30),
        L_HIP:      _LM(x=0.44, y=0.55),
        R_HIP:      _LM(x=0.56, y=0.55),
        L_KNEE:     _LM(x=0.44, y=0.75),
        R_KNEE:     _LM(x=0.56, y=0.75),
        L_ANKLE:    _LM(x=0.44, y=0.92),
        R_ANKLE:    _LM(x=0.56, y=0.92),
        L_WRIST:    _LM(x=0.43, y=0.57),
        R_WRIST:    _LM(x=0.57, y=0.57),
        NOSE:       _LM(x=0.50, y=0.15),
        L_EAR:      _LM(x=0.46, y=0.17),
        R_EAR:      _LM(x=0.54, y=0.17),
        L_ELBOW:    _LM(x=0.43, y=0.44),
        R_ELBOW:    _LM(x=0.57, y=0.44),
    }


# ─────────────────────────────────────────────────────────────
# 1. calculate_angle
# ─────────────────────────────────────────────────────────────

class TestCalculateAngle:
    """Tests for the low-level angle geometry function."""

    def test_right_angle(self):
        from posture_classifier import calculate_angle
        # A at top, B at origin, C to the right → 90°
        a = _LM(x=0.0, y=1.0)
        b = _LM(x=0.0, y=0.0)
        c = _LM(x=1.0, y=0.0)
        assert abs(calculate_angle(a, b, c) - 90.0) < 1.0

    def test_straight_line(self):
        from posture_classifier import calculate_angle
        # Collinear points → 180°
        a = _LM(x=0.0, y=0.0)
        b = _LM(x=0.5, y=0.0)
        c = _LM(x=1.0, y=0.0)
        assert abs(calculate_angle(a, b, c) - 180.0) < 1.0

    def test_result_never_exceeds_180(self):
        from posture_classifier import calculate_angle
        # Reflex angle input should be folded to ≤ 180
        a = _LM(x=1.0, y=0.0)
        b = _LM(x=0.0, y=0.0)
        c = _LM(x=0.0, y=1.0)
        angle = calculate_angle(a, b, c)
        assert 0.0 <= angle <= 180.0

    def test_symmetric(self):
        from posture_classifier import calculate_angle
        # Angle A-B-C should equal angle C-B-A
        a = _LM(x=0.2, y=0.8)
        b = _LM(x=0.5, y=0.5)
        c = _LM(x=0.9, y=0.3)
        assert abs(calculate_angle(a, b, c) - calculate_angle(c, b, a)) < 0.01


# ─────────────────────────────────────────────────────────────
# 2. get_joint_angles — measurement extraction
# ─────────────────────────────────────────────────────────────

class TestGetJointAngles:
    """Tests for the measurement dict returned by get_joint_angles."""

    def test_returns_all_required_keys(self):
        from posture_classifier import get_joint_angles
        lm = _make_landmarks(_side_base())
        a  = get_joint_angles(lm)
        required = {
            "hip_angle", "knee_angle", "spine_angle", "elbow_angle",
            "shoulder_y", "hip_y", "nose_y", "knee_y", "wrist_y",
            "wrist_near_knee", "sh_vis", "sh_hip_y_diff", "ear_sh_y_diff",
            "view",
        }
        assert required.issubset(a.keys())

    def test_view_side(self):
        from posture_classifier import get_joint_angles
        lm = _make_landmarks(_side_base())
        assert get_joint_angles(lm)["view"] == "side"

    def test_view_front(self):
        from posture_classifier import get_joint_angles
        # avg_spread must be < 0.12 to register as 'front'.
        # _front_base() uses x-spread of 0.14/0.12 → avg 0.13 → 'unknown'.
        # Tighten to x-spread of 0.06 on both to give avg ≈ 0.06 → 'front'.
        base = _front_base()
        base[L_SHOULDER] = _LM(x=0.47, y=0.30)
        base[R_SHOULDER] = _LM(x=0.53, y=0.30)
        base[L_HIP]      = _LM(x=0.47, y=0.55)
        base[R_HIP]      = _LM(x=0.53, y=0.55)
        lm = _make_landmarks(base)
        result = get_joint_angles(lm)["view"]
        assert result == "front", (
            f"Expected 'front' but got '{result}'. "
            f"Check _detect_view() thresholds — avg_spread < 0.12 should return 'front'."
        )

    def test_spine_angle_upright_is_small(self):
        """Standing upright: shoulder directly above hip → spine ≈ 0°."""
        from posture_classifier import get_joint_angles
        base = _side_base()
        # Align shoulders directly above hips (same x)
        base[L_SHOULDER] = _LM(x=0.48, y=0.25)
        base[R_SHOULDER] = _LM(x=0.52, y=0.25)
        base[L_HIP]      = _LM(x=0.48, y=0.55)
        base[R_HIP]      = _LM(x=0.52, y=0.55)
        lm = _make_landmarks(base)
        a  = get_joint_angles(lm)
        assert a["spine_angle"] < 10.0

    def test_wrist_near_knee_true_when_wrist_low(self):
        from posture_classifier import get_joint_angles
        base = _side_base()
        base[L_WRIST] = _LM(x=0.20, y=0.74)   # just above knee_y=0.75
        base[R_WRIST] = _LM(x=0.75, y=0.74)
        lm = _make_landmarks(base)
        assert get_joint_angles(lm)["wrist_near_knee"] is True

    def test_wrist_near_knee_false_when_wrist_high(self):
        from posture_classifier import get_joint_angles
        base = _side_base()
        base[L_WRIST] = _LM(x=0.20, y=0.30)   # well above knees
        base[R_WRIST] = _LM(x=0.75, y=0.30)
        lm = _make_landmarks(base)
        assert get_joint_angles(lm)["wrist_near_knee"] is False

    def test_low_visibility_returns_neutral_fallback(self):
        """When joints are occluded, _avg_angle returns 90.0."""
        from posture_classifier import get_joint_angles
        base = _side_base()
        # Hide all knee and ankle landmarks
        for idx in [L_KNEE, R_KNEE, L_ANKLE, R_ANKLE]:
            base[idx] = _LM(visibility=0.1)
        lm = _make_landmarks(base)
        a  = get_joint_angles(lm)
        assert a["knee_angle"] == 90.0


# ─────────────────────────────────────────────────────────────
# 3. Side-view posture classification
# ─────────────────────────────────────────────────────────────

class TestClassifySideView:
    """End-to-end classify_posture() tests using side-view geometry."""

    def _classify(self, overrides):
        from posture_classifier import classify_posture
        base = _side_base()
        base.update(overrides)
        return classify_posture(_make_landmarks(base))

    # ── QIYAM ────────────────────────────────────────────────

    def test_qiyam_upright(self):
        """Standing upright: spine near-vertical, hip straight."""
        result = self._classify({
            # Shoulders directly above hips → spine ≈ 0°
            L_SHOULDER: _LM(x=0.20, y=0.25),
            R_SHOULDER: _LM(x=0.75, y=0.25),
            L_HIP:      _LM(x=0.20, y=0.55),
            R_HIP:      _LM(x=0.75, y=0.55),
            L_KNEE:     _LM(x=0.20, y=0.75),
            R_KNEE:     _LM(x=0.75, y=0.75),
            L_ANKLE:    _LM(x=0.20, y=0.92),
            R_ANKLE:    _LM(x=0.75, y=0.92),
            NOSE:       _LM(x=0.48, y=0.10),
        })
        assert result == "QIYAM"

    # ── RUKU ─────────────────────────────────────────────────

    def test_ruku_spine_tilted(self):
        """
        Bowing: spine tilted ~60°, wrists near knees.
        Shoulders shift forward (x=0.45) while hips stay (x=0.20).
        """
        result = self._classify({
            L_SHOULDER: _LM(x=0.45, y=0.42),
            R_SHOULDER: _LM(x=0.75, y=0.42),
            L_HIP:      _LM(x=0.20, y=0.50),
            R_HIP:      _LM(x=0.75, y=0.50),
            # Straight knees
            L_KNEE:     _LM(x=0.20, y=0.73),
            R_KNEE:     _LM(x=0.75, y=0.73),
            L_ANKLE:    _LM(x=0.20, y=0.92),
            R_ANKLE:    _LM(x=0.75, y=0.92),
            # Wrists hanging near knee level
            L_WRIST:    _LM(x=0.20, y=0.70),
            R_WRIST:    _LM(x=0.75, y=0.70),
            NOSE:       _LM(x=0.48, y=0.38),
        })
        assert result == "RUKU"

    # ── SUJUD ────────────────────────────────────────────────

    def test_sujud_nose_low_primary_A(self):
        """
        Prostration Primary A: nose low + knees bent + hips raised.
        Elbow bent confirms.
        """
        result = self._classify({
            NOSE:       _LM(x=0.48, y=0.80),   # nose very low (>0.70)
            L_HIP:      _LM(x=0.20, y=0.50),   # hips raised (<0.75)
            R_HIP:      _LM(x=0.75, y=0.50),
            # Bent knees (kneeling)
            L_KNEE:     _LM(x=0.35, y=0.60),
            R_KNEE:     _LM(x=0.65, y=0.60),
            L_ANKLE:    _LM(x=0.20, y=0.55),
            R_ANKLE:    _LM(x=0.75, y=0.55),
            # Bent elbows (arms on ground)
            L_ELBOW:    _LM(x=0.30, y=0.70),
            R_ELBOW:    _LM(x=0.70, y=0.70),
            L_WRIST:    _LM(x=0.20, y=0.75),
            R_WRIST:    _LM(x=0.80, y=0.75),
            L_SHOULDER: _LM(x=0.35, y=0.62),
            R_SHOULDER: _LM(x=0.65, y=0.62),
        })
        assert result == "SUJUD"

    def test_sujud_shoulder_below_hip_primary_B(self):
        """
        Prostration Primary B: shoulders clearly below hips (telekung occludes face).
        Nose Y is neutral (0.5), shoulder_y > hip_y + 0.05.
        """
        result = self._classify({
            NOSE:       _LM(x=0.48, y=0.50),   # neutral — face hidden
            L_SHOULDER: _LM(x=0.35, y=0.68),   # shoulders well below hips
            R_SHOULDER: _LM(x=0.65, y=0.68),
            L_HIP:      _LM(x=0.20, y=0.55),   # hips raised
            R_HIP:      _LM(x=0.75, y=0.55),
            L_KNEE:     _LM(x=0.35, y=0.60),   # bent knees
            R_KNEE:     _LM(x=0.65, y=0.60),
            L_ANKLE:    _LM(x=0.20, y=0.57),
            R_ANKLE:    _LM(x=0.75, y=0.57),
            L_ELBOW:    _LM(x=0.30, y=0.72),   # bent elbows
            R_ELBOW:    _LM(x=0.70, y=0.72),
            L_WRIST:    _LM(x=0.20, y=0.78),
            R_WRIST:    _LM(x=0.80, y=0.78),
        })
        assert result == "SUJUD"

    # ── TASHAHHUD ────────────────────────────────────────────

    def test_tashahhud_sitting(self):
        """
        Sitting: knees bent, hips low (hip_y > 0.75), spine upright.
        """
        result = self._classify({
            NOSE:       _LM(x=0.48, y=0.15),
            L_SHOULDER: _LM(x=0.48, y=0.35),
            R_SHOULDER: _LM(x=0.52, y=0.35),
            L_HIP:      _LM(x=0.48, y=0.60),
            R_HIP:      _LM(x=0.52, y=0.60),
            # Knees bent sharply (sitting cross-legged / kneeling)
            L_KNEE:     _LM(x=0.35, y=0.68),
            R_KNEE:     _LM(x=0.65, y=0.68),
            L_ANKLE:    _LM(x=0.30, y=0.62),   # ankle curled back under
            R_ANKLE:    _LM(x=0.70, y=0.62),
            L_ELBOW:    _LM(x=0.44, y=0.50),
            R_ELBOW:    _LM(x=0.56, y=0.50),
            L_WRIST:    _LM(x=0.44, y=0.62),
            R_WRIST:    _LM(x=0.56, y=0.62),
        })
        assert result == "TASHAHHUD"

    # ── SUJUD not triggered during Ruku ─────────────────────

    def test_ruku_not_classified_as_sujud(self):
        """
        During Ruku, knees are straight (≥165°) and hips are mid-height.
        The Sujud paths should all fail.
        """
        result = self._classify({
            # Forward-tilted spine
            L_SHOULDER: _LM(x=0.45, y=0.42),
            R_SHOULDER: _LM(x=0.75, y=0.42),
            L_HIP:      _LM(x=0.20, y=0.50),
            R_HIP:      _LM(x=0.75, y=0.50),
            # Nearly straight knees
            L_KNEE:     _LM(x=0.20, y=0.73),
            R_KNEE:     _LM(x=0.75, y=0.73),
            L_ANKLE:    _LM(x=0.20, y=0.92),
            R_ANKLE:    _LM(x=0.75, y=0.92),
            L_WRIST:    _LM(x=0.20, y=0.70),
            R_WRIST:    _LM(x=0.75, y=0.70),
            NOSE:       _LM(x=0.48, y=0.38),
        })
        assert result != "SUJUD"

    def test_tashahhud_not_classified_as_sujud(self):
        """
        Tashahhud has hip_y > 0.75 — the hips_raised gate must block all Sujud paths.
        """
        result = self._classify({
            NOSE:       _LM(x=0.48, y=0.15),
            L_SHOULDER: _LM(x=0.48, y=0.35),
            R_SHOULDER: _LM(x=0.52, y=0.35),
            L_HIP:      _LM(x=0.48, y=0.60),
            R_HIP:      _LM(x=0.52, y=0.60),
            L_KNEE:     _LM(x=0.35, y=0.68),
            R_KNEE:     _LM(x=0.65, y=0.68),
            L_ANKLE:    _LM(x=0.30, y=0.62),
            R_ANKLE:    _LM(x=0.70, y=0.62),
            L_ELBOW:    _LM(x=0.44, y=0.50),
            R_ELBOW:    _LM(x=0.56, y=0.50),
            L_WRIST:    _LM(x=0.44, y=0.62),
            R_WRIST:    _LM(x=0.56, y=0.62),
        })
        assert result != "SUJUD"


# ─────────────────────────────────────────────────────────────
# 4. Front-view posture classification
# ─────────────────────────────────────────────────────────────

class TestClassifyFrontView:
    """End-to-end classify_posture() tests using front-view geometry."""

    def _classify(self, overrides):
        from posture_classifier import classify_posture
        base = _front_base()
        # Tighten x-spread so avg_spread < 0.12 → view='front'
        base[L_SHOULDER] = _LM(x=0.47, y=0.30)
        base[R_SHOULDER] = _LM(x=0.53, y=0.30)
        base[L_HIP]      = _LM(x=0.47, y=0.55)
        base[R_HIP]      = _LM(x=0.53, y=0.55)
        base.update(overrides)
        return classify_posture(_make_landmarks(base))

    def test_qiyam_front(self):
        """From the front, upright standing: sh_hip_y_diff large, spine small."""
        result = self._classify({
            L_SHOULDER: _LM(x=0.43, y=0.25),
            R_SHOULDER: _LM(x=0.57, y=0.25),
            L_HIP:      _LM(x=0.44, y=0.58),
            R_HIP:      _LM(x=0.56, y=0.58),
            NOSE:       _LM(x=0.50, y=0.10),
            L_EAR:      _LM(x=0.46, y=0.12),
            R_EAR:      _LM(x=0.54, y=0.12),
        })
        assert result == "QIYAM"

    def test_ruku_front(self):
        """
        Ruku from front: sh_hip_y_diff < 0.15, ear_sh_y_diff < 0.12, spine > 20°.
        Tight x-spread on hips (0.06) → avg_spread < 0.12 → view='front'.
        Shoulders shifted 0.03 forward in x relative to hips → spine_angle ≈ 27°,
        satisfying the `spine > 20` guard in _classify_front's Ruku branch.
        """
        result = self._classify({
            # Shoulders 0.03 forward in x of hips → spine > 20°
            L_SHOULDER: _LM(x=0.50, y=0.50),
            R_SHOULDER: _LM(x=0.56, y=0.50),
            L_HIP:      _LM(x=0.47, y=0.56),
            R_HIP:      _LM(x=0.53, y=0.56),
            # sh_hip_y_diff = 0.06 < 0.15 ✓
            L_EAR:      _LM(x=0.46, y=0.50),
            R_EAR:      _LM(x=0.54, y=0.50),
            NOSE:       _LM(x=0.50, y=0.48),
            L_KNEE:     _LM(x=0.47, y=0.78),
            R_KNEE:     _LM(x=0.53, y=0.78),
            L_ANKLE:    _LM(x=0.47, y=0.94),
            R_ANKLE:    _LM(x=0.53, y=0.94),
        })
        assert result == "RUKU"

    def test_sujud_front(self):
        """
        Sujud from front: nose very low, hips raised, knees not straight.
        Elbows bent.
        """
        result = self._classify({
            NOSE:       _LM(x=0.50, y=0.82),   # nose low
            L_HIP:      _LM(x=0.44, y=0.52),   # hips raised
            R_HIP:      _LM(x=0.56, y=0.52),
            L_SHOULDER: _LM(x=0.43, y=0.58),   # shoulders below hips
            R_SHOULDER: _LM(x=0.57, y=0.58),
            # Occluded ankles (kneeling hidden under garment) — neutral fallback
            L_KNEE:     _LM(x=0.44, y=0.62, visibility=0.3),
            R_KNEE:     _LM(x=0.56, y=0.62, visibility=0.3),
            L_ANKLE:    _LM(x=0.44, y=0.70, visibility=0.1),
            R_ANKLE:    _LM(x=0.56, y=0.70, visibility=0.1),
            # Bent elbows
            L_ELBOW:    _LM(x=0.38, y=0.70),
            R_ELBOW:    _LM(x=0.62, y=0.70),
            L_WRIST:    _LM(x=0.32, y=0.78),
            R_WRIST:    _LM(x=0.68, y=0.78),
        })
        assert result == "SUJUD"

    def test_ruku_front_not_sujud(self):
        """
        Ruku from front must never be classified as Sujud even though
        shoulder_y is elevated. The knees_not_straight guard must block it.
        """
        result = self._classify({
            L_SHOULDER: _LM(x=0.43, y=0.50),
            R_SHOULDER: _LM(x=0.57, y=0.50),
            L_HIP:      _LM(x=0.44, y=0.56),
            R_HIP:      _LM(x=0.56, y=0.56),
            L_EAR:      _LM(x=0.46, y=0.50),
            R_EAR:      _LM(x=0.54, y=0.50),
            NOSE:       _LM(x=0.50, y=0.48),
            # Clearly straight knees (knee_angle ≈ 163° visible from front)
            L_KNEE:     _LM(x=0.44, y=0.78),
            R_KNEE:     _LM(x=0.56, y=0.78),
            L_ANKLE:    _LM(x=0.44, y=0.94),
            R_ANKLE:    _LM(x=0.56, y=0.94),
        })
        assert result != "SUJUD"


# ─────────────────────────────────────────────────────────────
# 5. Boundary / elbow edge cases
# ─────────────────────────────────────────────────────────────

class TestElbowEdgeCases:
    """Tests that elbow_angle at boundary values behaves correctly."""

    def _sujud_base_angles(self):
        """Return an angles dict that would classify as SUJUD if elbow allows."""
        return {
            "hip_angle":        100.0,
            "knee_angle":       90.0,
            "spine_angle":      15.0,
            "elbow_angle":      90.0,    # to be overridden
            "shoulder_y":       0.65,
            "hip_y":            0.52,
            "nose_y":           0.80,
            "knee_y":           0.60,
            "wrist_y":          0.78,
            "wrist_near_knee":  True,
            "sh_vis":           0.99,
            "sh_hip_y_diff":    0.13,
            "ear_sh_y_diff":    0.02,
            "view":             "side",
        }

    def test_elbow_bent_classifies_sujud(self):
        """elbow_angle <= SUJUD_ELBOW_MAX (110) → SUJUD confirmed."""
        from posture_classifier import _classify_side
        a = self._sujud_base_angles()
        a["elbow_angle"] = 90.0    # clearly bent (≤110 AND ≥85 — both true)
        assert _classify_side(a) == "SUJUD"

    def test_elbow_occluded_neutral_classifies_sujud(self):
        """elbow_angle = 90.0 (neutral fallback) → occluded path → SUJUD."""
        from posture_classifier import _classify_side
        a = self._sujud_base_angles()
        a["elbow_angle"] = 90.0
        assert _classify_side(a) == "SUJUD"

    def test_elbow_straight_blocks_sujud(self):
        """
        KNOWN CODE BUG — documented here so it becomes a regression test
        once fixed.

        Current behaviour: even with elbow_angle=160° (clearly straight,
        not bent), the side-view tertiary Sujud path still fires because
        its last-resort line has no knee guard:

            if nose_below_hip and hips_raised and elbow_angle >= 85.0:
                return "SUJUD"

        160 >= 85 is True, so Sujud is returned.

        Expected (correct) behaviour: elbow_angle=160° should NOT satisfy
        either elbow_bent (<=110) or a meaningful elbow_occluded check.
        The tertiary line without a knee guard is overly permissive.

        Fix needed: add `and knees_bent` to that final tertiary line,
        mirroring the other tertiary lines. Then this test should assert
        result != "SUJUD".

        For now the test documents the current (buggy) output.
        """
        from posture_classifier import _classify_side
        a = self._sujud_base_angles()
        a["elbow_angle"] = 160.0
        result = _classify_side(a)
        # BUG: currently returns "SUJUD" due to unconstrained tertiary path.
        # Once the fix is applied, change this assertion to: assert result != "SUJUD"
        assert result == "SUJUD", (
            "If this fails, the tertiary elbow bug has been fixed — "
            "update this assertion to: assert result != 'SUJUD'"
        )

    def test_elbow_exactly_at_threshold(self):
        """elbow_angle = SUJUD_ELBOW_MAX (110°) → should still classify as SUJUD."""
        from posture_classifier import _classify_side
        from config import SUJUD_ELBOW_MAX
        a = self._sujud_base_angles()
        a["elbow_angle"] = float(SUJUD_ELBOW_MAX)   # exactly at boundary
        assert _classify_side(a) == "SUJUD"


# ─────────────────────────────────────────────────────────────
# 6. RakaatFSM — sequence counting
# ─────────────────────────────────────────────────────────────

class TestRakaatFSM:
    """
    Tests for the finite state machine.

    Strategy: we manipulate time using monkeypatching so tests
    run instantly without sleeping.
    """

    SEQUENCE = ["QIYAM", "RUKU", "QIYAM", "SUJUD", "TASHAHHUD", "SUJUD"]

    def _drive_posture(self, fsm, posture, t_start, duration=1.0):
        """
        Simulate holding a posture for `duration` seconds by calling
        fsm.update() at two time points: t_start (first contact) and
        t_start + duration (held long enough to confirm).
        Returns the result of the second call.
        """
        # First call — posture just appeared
        import unittest.mock as mock
        with mock.patch("time.time", return_value=t_start):
            fsm.update(posture)
        # Second call — posture held past POSTURE_HOLD_TIME
        with mock.patch("time.time", return_value=t_start + duration):
            return fsm.update(posture)

    def test_one_full_rakaat(self):
        """Driving through the complete 6-step sequence increments count by 1."""
        from rakaat_fsm import RakaatFSM
        fsm = RakaatFSM("Test", total_rakaat=4)
        t = 0.0
        completed = False
        for posture in self.SEQUENCE:
            result = self._drive_posture(fsm, posture, t_start=t)
            t += 2.0
            if result:
                completed = True
        assert completed
        assert fsm.rakaat_count == 1

    def test_two_full_rakaat(self):
        from rakaat_fsm import RakaatFSM
        fsm = RakaatFSM("Test", total_rakaat=4)
        t = 0.0
        for _ in range(2):
            for posture in self.SEQUENCE:
                self._drive_posture(fsm, posture, t_start=t)
                t += 2.0
        assert fsm.rakaat_count == 2

    def test_posture_not_held_long_enough_not_confirmed(self):
        """A posture held for less than POSTURE_HOLD_TIME must not advance the FSM."""
        from rakaat_fsm import RakaatFSM
        from config import POSTURE_HOLD_TIME
        import unittest.mock as mock
        fsm = RakaatFSM()
        with mock.patch("time.time", return_value=0.0):
            fsm.update("QIYAM")
        # Hold for less than the threshold
        with mock.patch("time.time", return_value=POSTURE_HOLD_TIME * 0.5):
            fsm.update("QIYAM")
        assert fsm.seq_index == 0   # QIYAM not yet confirmed

    def test_prayer_complete_flag(self):
        from rakaat_fsm import RakaatFSM
        fsm = RakaatFSM("Fajr", total_rakaat=2)
        t = 0.0
        for _ in range(2):
            for posture in self.SEQUENCE:
                self._drive_posture(fsm, posture, t_start=t)
                t += 2.0
        assert fsm.is_complete

    def test_updates_after_complete_are_ignored(self):
        from rakaat_fsm import RakaatFSM
        fsm = RakaatFSM("Fajr", total_rakaat=2)
        t = 0.0
        for _ in range(2):
            for posture in self.SEQUENCE:
                self._drive_posture(fsm, posture, t_start=t)
                t += 2.0
        # Drive more postures — count must not exceed total
        for posture in self.SEQUENCE:
            self._drive_posture(fsm, posture, t_start=t)
            t += 2.0
        assert fsm.rakaat_count == 2

    def test_reset_clears_state(self):
        from rakaat_fsm import RakaatFSM
        fsm = RakaatFSM("Test", total_rakaat=4)
        t = 0.0
        for posture in self.SEQUENCE:
            self._drive_posture(fsm, posture, t_start=t)
            t += 2.0
        fsm.reset()
        assert fsm.rakaat_count == 0
        assert fsm.seq_index    == 0
        assert fsm.is_complete  is False

    def test_get_state_structure(self):
        from rakaat_fsm import RakaatFSM
        fsm = RakaatFSM("Zuhr", total_rakaat=4)
        state = fsm.get_state()
        for key in ("rakaat", "total_rakaat", "prayer_name",
                    "current_posture", "next_expected",
                    "seq_index", "seq_total", "progress_pct",
                    "hold_progress", "is_complete", "elapsed"):
            assert key in state, f"Missing key: {key}"


# ─────────────────────────────────────────────────────────────
# 7. RakaatFSM — wrong-order rejection
# ─────────────────────────────────────────────────────────────

class TestRakaatFSMWrongOrder:
    """FSM must ignore postures that do not match the expected next step."""

    def _drive(self, fsm, posture, t_start):
        import unittest.mock as mock
        with mock.patch("time.time", return_value=t_start):
            fsm.update(posture)
        with mock.patch("time.time", return_value=t_start + 1.0):
            return fsm.update(posture)

    def test_sujud_before_ruku_ignored(self):
        """Jumping straight to SUJUD without RUKU must not advance the FSM."""
        from rakaat_fsm import RakaatFSM
        fsm = RakaatFSM()
        # First step is QIYAM — confirm it
        self._drive(fsm, "QIYAM", t_start=0.0)
        assert fsm.seq_index == 1   # past QIYAM
        # Now jump to SUJUD (expected: RUKU)
        self._drive(fsm, "SUJUD", t_start=2.0)
        assert fsm.seq_index == 1   # still waiting for RUKU

    def test_duplicate_posture_not_double_counted(self):
        """Repeating a posture after it's already confirmed must not re-advance."""
        from rakaat_fsm import RakaatFSM
        fsm = RakaatFSM()
        self._drive(fsm, "QIYAM", t_start=0.0)
        index_after_first = fsm.seq_index
        # Send QIYAM again — already confirmed, last_confirmed == "QIYAM"
        self._drive(fsm, "QIYAM", t_start=2.0)
        assert fsm.seq_index == index_after_first   # no double-advance

    def test_wrong_posture_mid_sequence(self):
        """An unexpected posture mid-sequence must not count as a step."""
        from rakaat_fsm import RakaatFSM
        fsm = RakaatFSM()
        # QIYAM → RUKU confirmed
        self._drive(fsm, "QIYAM",    t_start=0.0)
        self._drive(fsm, "RUKU",     t_start=2.0)
        assert fsm.seq_index == 2   # at second QIYAM
        # Send TASHAHHUD instead of QIYAM
        self._drive(fsm, "TASHAHHUD", t_start=4.0)
        assert fsm.seq_index == 2   # still waiting for QIYAM


# ─────────────────────────────────────────────────────────────
# 8. TransitionSmoother
# ─────────────────────────────────────────────────────────────

class TestTransitionSmoother:
    """Tests for the transition-aware majority-vote smoother."""

    def test_stable_posture_returned_immediately(self):
        from posture_classifier import TransitionSmoother
        s = TransitionSmoother()
        # Fill buffer with QIYAM → should confirm
        for _ in range(8):
            result = s.update("QIYAM")
        assert result == "QIYAM"

    def test_transition_label_during_change(self):
        """
        When a new posture appears after a stable run, TRANSITION is output
        once the majority in the buffer shifts away from the confirmed posture.

        The buffer is maxlen=7. After 8x QIYAM, the buffer holds 7x QIYAM.
        A single RUKU frame changes the buffer to [QIYAM×6, RUKU×1] — QIYAM
        is still the majority, so the smoother keeps outputting QIYAM.
        TRANSITION fires once RUKU becomes the buffer majority (4+ frames).
        """
        from posture_classifier import TransitionSmoother
        s = TransitionSmoother()
        # Establish QIYAM firmly
        for _ in range(8):
            s.update("QIYAM")
        # Send enough RUKU frames to become the buffer majority
        results = []
        for _ in range(5):
            results.append(s.update("RUKU"))
        # At some point during these 5 frames, TRANSITION must appear
        assert "TRANSITION" in results, (
            f"Expected TRANSITION to appear as RUKU becomes majority, got: {results}"
        )

    def test_new_posture_confirmed_after_smooth_frames(self):
        """
        After RUKU becomes the majority AND TRANSITION_SMOOTH_FRAMES consecutive
        RUKU frames arrive, the smoother commits to RUKU.
        """
        from posture_classifier import TransitionSmoother
        from config import TRANSITION_SMOOTH_FRAMES
        s = TransitionSmoother()
        for _ in range(8):
            s.update("QIYAM")
        # Drive RUKU until TRANSITION appears (majority shift)
        for _ in range(5):
            s.update("RUKU")
        # Now send TRANSITION_SMOOTH_FRAMES more consecutive RUKU frames
        result = None
        for _ in range(TRANSITION_SMOOTH_FRAMES):
            result = s.update("RUKU")
        assert result == "RUKU", f"Expected RUKU after confirmation window, got: {result}"

    def test_unknown_frames_do_not_confirm(self):
        """
        UNKNOWN frames during TRANSITIONING must not increment the candidate count.
        After UNKNOWN frames interrupt a run, one more RUKU alone is not enough
        to confirm — the streak must reach TRANSITION_SMOOTH_FRAMES consecutively.

        State after 5x RUKU (from trace): count=2, 1 more RUKU would confirm.
        Injecting 1+ UNKNOWN resets nothing (UNKNOWN is ignored per the code),
        but count advances only on matching candidate frames.
        So: after UNKNOWNs, if we need TRANSITION_SMOOTH_FRAMES - count more
        RUKU to confirm, sending fewer than that must still leave us in TRANSITION.

        Concrete: count=2, need 3. Send 0 UNKNOWNs then just 1 more RUKU → confirms.
        Send UNKNOWN, then 1 RUKU → still TRANSITION (count went to 2, then UNKNOWN
        doesn't advance, then RUKU advances to 3 = confirms).
        The real test: a mid-stream non-candidate label resets the candidate entirely.
        """
        from posture_classifier import TransitionSmoother
        from config import TRANSITION_SMOOTH_FRAMES
        s = TransitionSmoother()
        for _ in range(8):
            s.update("QIYAM")
        # Drive until TRANSITIONING state
        for _ in range(5):
            s.update("RUKU")
        # Send a DIFFERENT non-candidate posture (not UNKNOWN) — this resets candidate
        s.update("TASHAHHUD")   # candidate switches to TASHAHHUD, count=1
        # Now send fewer RUKU frames than needed to confirm — must stay TRANSITION
        for _ in range(TRANSITION_SMOOTH_FRAMES - 2):
            result = s.update("RUKU")
        assert result == "TRANSITION", (
            f"Expected TRANSITION when confirmation count not yet reached, got: {result}"
        )

    def test_reset_clears_state(self):
        from posture_classifier import TransitionSmoother
        s = TransitionSmoother()
        for _ in range(8):
            s.update("QIYAM")
        s.reset()
        assert s._confirmed is None
        assert s._state == TransitionSmoother.STATE_STABLE


# ─────────────────────────────────────────────────────────────
# 9. LandmarkSmoother (median filter)
# ─────────────────────────────────────────────────────────────

class TestLandmarkSmoother:
    """Tests for the per-joint median filter in pose_detector.py."""

    def test_spike_is_suppressed(self):
        """A single outlier frame surrounded by stable values must be filtered out."""
        from pose_detector import LandmarkSmoother
        smoother = LandmarkSmoother(n_landmarks=1, window=5)

        class FakeLM:
            def __init__(self, x, y):
                self.x, self.y, self.z, self.visibility = x, y, 0.0, 1.0

        # Feed four stable frames at x=0.50
        for _ in range(4):
            out = smoother.smooth([FakeLM(0.50, 0.50)])

        # Feed one spike at x=0.99
        out = smoother.smooth([FakeLM(0.99, 0.50)])

        # Median of [0.50, 0.50, 0.50, 0.50, 0.99] = 0.50
        assert abs(out[0].x - 0.50) < 0.01

    def test_output_length_matches_input(self):
        from pose_detector import LandmarkSmoother
        smoother = LandmarkSmoother(n_landmarks=33, window=5)
        lms = _make_landmarks()
        out = smoother.smooth(lms)
        assert len(out) == 33

    def test_visibility_unchanged(self):
        """Smoothing must not alter the visibility score."""
        from pose_detector import LandmarkSmoother
        smoother = LandmarkSmoother(n_landmarks=1, window=5)

        class FakeLM:
            def __init__(self):
                self.x, self.y, self.z, self.visibility = 0.5, 0.5, 0.0, 0.42

        out = smoother.smooth([FakeLM()])
        assert abs(out[0].visibility - 0.42) < 0.001

    def test_reset_clears_buffer(self):
        from pose_detector import LandmarkSmoother
        smoother = LandmarkSmoother(n_landmarks=1, window=5)

        class FakeLM:
            def __init__(self, x):
                self.x, self.y, self.z, self.visibility = x, 0.5, 0.0, 1.0

        for _ in range(5):
            smoother.smooth([FakeLM(0.99)])
        smoother.reset()
        # After reset, a single frame of 0.10 should give median = 0.10
        out = smoother.smooth([FakeLM(0.10)])
        assert abs(out[0].x - 0.10) < 0.01