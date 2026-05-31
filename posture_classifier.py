# ============================================================
# posture_classifier.py — Posture classification
#
# Implements professor directives:
#   #1 — Front-view support: detects Ruku from the front using
#        shoulder-to-hip Y proximity (depth collapse workaround)
#   #2 — Transition smoothing: TransitionSmoother applies
#        weighted smoothing specifically during posture changes,
#        not just within stable postures
#   Telekung robustness retained from previous rewrite:
#        nose-based Sujud, spine-tilt Ruku, visibility gating
# ============================================================

import numpy as np
from collections import deque
import mediapipe as mp
from config import (
    QIYAM_HIP_MIN,
    RUKU_HIP_MAX, RUKU_HIP_MIN,
    SUJUD_SHOULDER_HIP_DIFF, SUJUD_ELBOW_MAX,
    TASHAHHUD_KNEE_MAX, TASHAHHUD_HIP_MAX,
    SMOOTHING_WINDOW,
    VISIBILITY_THRESHOLD,
    TRANSITION_SMOOTH_FRAMES,
)

# ── Landmark indices ─────────────────────────────────────────
mp_pose = mp.solutions.pose
PL = mp_pose.PoseLandmark

L_SHOULDER = PL.LEFT_SHOULDER.value
R_SHOULDER = PL.RIGHT_SHOULDER.value
L_HIP      = PL.LEFT_HIP.value
R_HIP      = PL.RIGHT_HIP.value
L_KNEE     = PL.LEFT_KNEE.value
R_KNEE     = PL.RIGHT_KNEE.value
L_ANKLE    = PL.LEFT_ANKLE.value
R_ANKLE    = PL.RIGHT_ANKLE.value
L_WRIST    = PL.LEFT_WRIST.value
R_WRIST    = PL.RIGHT_WRIST.value
NOSE       = PL.NOSE.value
L_EAR      = PL.LEFT_EAR.value
R_EAR      = PL.RIGHT_EAR.value
L_ELBOW    = PL.LEFT_ELBOW.value
R_ELBOW    = PL.RIGHT_ELBOW.value


def _vis(lm, idx):
    """Return landmark visibility score, defaulting to 1.0 if unavailable."""
    try:
        return lm[idx].visibility
    except AttributeError:
        return 1.0


def calculate_angle(a, b, c):
    """
    Angle at joint B formed by A-B-C using arctan2.
    More stable than arccos — never produces NaN.
    """
    radians = (
        np.arctan2(c.y - b.y, c.x - b.x)
        - np.arctan2(a.y - b.y, a.x - b.x)
    )
    angle = np.abs(np.degrees(radians))
    if angle > 180.0:
        angle = 360.0 - angle
    return angle


def _avg_angle(lm, a1, b1, c1, a2, b2, c2,
               require_vis_a=None, require_vis_b=None):
    """
    Average angle from left and right sides.
    Skips a side if its landmarks are below the visibility threshold.
    Returns 90.0 if both sides are unreliable (neutral fallback).
    """
    left_ok  = (require_vis_a is None or
                all(_vis(lm, i) >= VISIBILITY_THRESHOLD for i in require_vis_a))
    right_ok = (require_vis_b is None or
                all(_vis(lm, i) >= VISIBILITY_THRESHOLD for i in require_vis_b))

    angles = []
    if left_ok:
        angles.append(calculate_angle(lm[a1], lm[b1], lm[c1]))
    if right_ok:
        angles.append(calculate_angle(lm[a2], lm[b2], lm[c2]))

    return sum(angles) / len(angles) if angles else 90.0


# ── View detection ───────────────────────────────────────────

def _detect_view(lm):
    """
    Heuristic to detect whether the camera is placed to the side
    or in front of the person.

    Side view:   left and right landmarks are spread horizontally
                 (e.g. left hip X much smaller than right hip X)
    Front view:  left and right landmarks are close in X
                 (both hips appear roughly centred)

    Returns: "side" | "front" | "unknown" 
    """
    l_hip_x = lm[L_HIP].x
    r_hip_x = lm[R_HIP].x
    l_sh_x  = lm[L_SHOULDER].x
    r_sh_x  = lm[R_SHOULDER].x

    hip_spread  = abs(r_hip_x - l_hip_x)
    sh_spread   = abs(r_sh_x  - l_sh_x)
    avg_spread  = (hip_spread + sh_spread) / 2

    # From the side, one hip/shoulder is far left and the other far right
    # From the front, both are closer together relative to frame width
    # Thresholds widened to treat 45° camera placement as "side":
    # At true 90° side view avg_spread ≈ 0.45–0.55
    # At 45° diagonal avg_spread ≈ 0.18–0.25
    # At true front view avg_spread ≈ 0.06–0.10
    # Lowering the side threshold from 0.25 → 0.18 routes 45° frames
    # to the side classifier instead of the unreliable fusion path.
# if avg_spread > 0.25:
#return "side"
#elif avg_spread < 0.12:
# return "front"
# else:
# return "unknown"

    if avg_spread > 0.18:
        return "side"
    return "front"



# ── Angle/measurement extraction ─────────────────────────────

def get_joint_angles(lm):
    """
    Compute all measurements used by both the side and front classifiers.
    Returns a dict consumed by classify_posture() and get_debug_angles().
    """
    # Hip angle — torso/hip/thigh
    hip_angle = _avg_angle(
        lm,
        L_SHOULDER, L_HIP, L_KNEE,
        R_SHOULDER, R_HIP, R_KNEE,
        require_vis_a=[L_SHOULDER, L_HIP, L_KNEE],
        require_vis_b=[R_SHOULDER, R_HIP, R_KNEE],
    )

    # Knee angle
    knee_angle = _avg_angle(
        lm,
        L_HIP, L_KNEE, L_ANKLE,
        R_HIP, R_KNEE, R_ANKLE,
        require_vis_a=[L_HIP, L_KNEE, L_ANKLE],
        require_vis_b=[R_HIP, R_KNEE, R_ANKLE],
    )

    # Spine tilt — angle of hip→shoulder vector from vertical
    sh_x  = (lm[L_SHOULDER].x + lm[R_SHOULDER].x) / 2
    sh_y  = (lm[L_SHOULDER].y + lm[R_SHOULDER].y) / 2
    hip_x = (lm[L_HIP].x + lm[R_HIP].x) / 2
    hip_y = (lm[L_HIP].y + lm[R_HIP].y) / 2

    dx = sh_x - hip_x
    dy = sh_y - hip_y
    spine_angle = abs(np.degrees(np.arctan2(abs(dx), abs(dy))))

    # Y positions (MediaPipe: 0=top, 1=bottom)
    nose_y  = lm[NOSE].y
    knee_y  = (lm[L_KNEE].y + lm[R_KNEE].y) / 2
    wrist_y = (lm[L_WRIST].y + lm[R_WRIST].y) / 2

    # Wrist near knee — true in Ruku (hands hang toward knees)
    wrist_near_knee = wrist_y >= knee_y - 0.08

    # ── Front-view specific measurements ─────────────────────
    # In front view, Ruku is hard to detect via angle alone because
    # depth is collapsed. Instead we check:
    #   (a) shoulder Y is close to hip Y — torso is nearly horizontal
    #   (b) ear Y is close to shoulder Y — head is at torso level
    sh_hip_y_diff   = abs(sh_y - hip_y)   # small = bowing from front
    ear_y           = (lm[L_EAR].y + lm[R_EAR].y) / 2
    ear_sh_y_diff   = abs(ear_y - sh_y)   # small = head level with shoulders

    # Shoulder visibility (for occlusion reporting)
    sh_vis = (_vis(lm, L_SHOULDER) + _vis(lm, R_SHOULDER)) / 2

    # Elbow angle — shoulder/elbow/wrist
    # In Sujud, elbows are bent (~60–100°) as arms support the body on the ground.
    # This is borrowed from Mughal's reference project as an additional Sujud signal.
    elbow_angle = _avg_angle(
        lm,
        L_SHOULDER, L_ELBOW, L_WRIST,
        R_SHOULDER, R_ELBOW, R_WRIST,
        require_vis_a=[L_SHOULDER, L_ELBOW, L_WRIST],
        require_vis_b=[R_SHOULDER, R_ELBOW, R_WRIST],
    )

    return {
        "hip_angle":        round(hip_angle, 1),
        "knee_angle":       round(knee_angle, 1),
        "spine_angle":      round(spine_angle, 1),
        "elbow_angle":      round(elbow_angle, 1),
        "shoulder_y":       round(sh_y, 3),
        "hip_y":            round(hip_y, 3),
        "nose_y":           round(nose_y, 3),
        "knee_y":           round(knee_y, 3),
        "wrist_y":          round(wrist_y, 3),
        "wrist_near_knee":  wrist_near_knee,
        "sh_vis":           round(sh_vis, 2),
        "sh_hip_y_diff":    round(sh_hip_y_diff, 3),
        "ear_sh_y_diff":    round(ear_sh_y_diff, 3),
        "view":             _detect_view(lm),
    }


# ── Side-view classifier ──────────────────────────────────────

def _classify_side(a):
    """
    Classify posture from side-view measurements.
    Uses spine tilt and nose position — robust for telekung.
    """
    hip   = a["hip_angle"]
    knee  = a["knee_angle"]
    spine = a["spine_angle"]
    nose_y = a["nose_y"]
    hip_y  = a["hip_y"]
    wrist_near_knee = a["wrist_near_knee"]

    # ── SUJUD detection — three paths in priority order ──────────
    #
    # Primary A: nose low + knees bent + hips raised
    #   → Use when face is visible (most common case without full telekung)
    #
    # Primary B: shoulder below hip + knees bent + hips raised
    #   → Use when face is hidden (telekung covers head during Sujud)
    #   → Geometrically reliable: hips being higher than shoulders is the
    #     defining physical characteristic of Sujud regardless of clothing
    #   → Only requires 4 landmarks (L/R shoulder + L/R hip midpoints)
    #   → Promoted from secondary because it is MORE reliable than nose_y
    #     which can be a neutral fallback or interpolated value
    #
    # Tertiary: nose below hip + hips raised
    #   → Weakest signal — demoted because nose_y is unreliable when face
    #     is occluded and may just reflect interpolated landmark positions
    #
    # All paths require hips_raised (hip_y < 0.75) to prevent Tashahhud
    # from being misclassified (sitting puts hip_y ~0.8).
    # All primary/B paths require knees_bent to prevent Ruku false positives
    # (Ruku has nearly straight knees ~165°).

    #
    # Side-view threshold notes (differ from front-view):
    #   knees_bent         = knee <= 110  (strict — side view sees the true angle)
    #   shoulder_below_hip margin = 0.03  (relaxed from 0.05 for 45° camera —
    #     at 45° diagonal the vertical separation between shoulder and hip is
    #     compressed, so the 0.05 margin was frequently not reached during Sujud)
    #   knees_ok_tertiary  = True always  (side can see knees; tertiary fires freely)

    nose_low           = nose_y > 0.70
    knees_bent         = knee <= 110
    nose_below_hip     = nose_y > hip_y + 0.05
    shoulder_below_hip = a["shoulder_y"] > a["hip_y"] + 0.03   # relaxed from 0.05
    hips_raised        = hip_y < 0.75
    elbow_bent         = a["elbow_angle"] <= SUJUD_ELBOW_MAX

    # Primary A — nose-based (face visible)
    primary_A = nose_low and knees_bent and hips_raised
    if primary_A and elbow_bent:
        return "SUJUD"
    if primary_A and a["elbow_angle"] >= 85.0:
        return "SUJUD"

    # Primary B — shoulder-based (face hidden, always geometrically valid)
    primary_B = shoulder_below_hip and knees_bent and hips_raised
    if primary_B and elbow_bent:
        return "SUJUD"
    if primary_B and a["elbow_angle"] >= 85.0:
        return "SUJUD"

    # Tertiary — nose below hip (weakest, face unreliable, last resort)
    if nose_below_hip and hips_raised and elbow_bent:
        return "SUJUD"
    if nose_below_hip and hips_raised and a["elbow_angle"] >= 85.0:
        return "SUJUD"

    # 2. TASHAHHUD — sitting: knee + hip bent, spine upright
    if knee <= TASHAHHUD_KNEE_MAX and hip <= TASHAHHUD_HIP_MAX and spine < 50:
        return "TASHAHHUD"

    # 3. RUKU — spine tilted 25–90° + wrists near knees or hip in range
    # Lower bound relaxed from 35° → 25° for 45° camera placement.
    # At true side view a full bow reads ~55–70°.
    # At 45° diagonal that same bow reads ~35–50° due to foreshortening.
    # Lowering to 25° ensures the bottom of the range is still caught.
    # Upper bound 90° unchanged — Tashahhud requires spine < 50° so
    # there is no overlap risk at the top.
    # Qiyam requires spine < 30° AND large hip angle — the hip guard
    # (wrist_near_knee or ruku_hip) prevents Qiyam from matching here.
    ruku_spine = 25 <= spine <= 90
    ruku_hip   = RUKU_HIP_MIN <= hip <= RUKU_HIP_MAX
    if ruku_spine and (wrist_near_knee or ruku_hip):
        return "RUKU"

    # 4. QIYAM — upright: spine near vertical, hip straight
    if spine < 30 and hip >= QIYAM_HIP_MIN:
        return "QIYAM"
    if spine < 25 and hip >= 140:
        return "QIYAM"

    return "UNKNOWN"


# ── Front-view classifier ─────────────────────────────────────

def _classify_front(a):
    """
    Classify posture from front-view measurements.

    From the front, depth is collapsed so angles are less reliable.
    We rely on Y-coordinate proximity between body parts instead.

    Key observations from front view:
      QIYAM:    shoulders well above hips; spine nearly vertical
      RUKU:     shoulders drop toward hip level (sh_hip_y_diff small)
                ear descends to shoulder level (head bows forward)
      SUJUD:    nose Y very low; knees bent
      TASHAHHUD:knee bent, hip bent (sitting — still detectable from front)
    """
    knee  = a["knee_angle"]
    hip   = a["hip_angle"]
    spine = a["spine_angle"]
    nose_y = a["nose_y"]
    hip_y  = a["hip_y"]
    sh_hip_y_diff = a["sh_hip_y_diff"]
    ear_sh_y_diff = a["ear_sh_y_diff"]

    # ── SUJUD detection — front view ─────────────────────────────
    # Bug fixed: During Ruku from the front, shoulder_y and hip_y are
    # both ~0.4. Frame-to-frame fluctuation makes shoulder_y occasionally
    # read 0.41 vs hip_y 0.40, triggering shoulder_below_hip with the
    # old 0.05 threshold. Two fixes:
    #   1. Tightened shoulder_below_hip from 0.05 to 0.10
    #   2. Added knees_not_straight guard — knee_angle >= 140 means
    #      knees are visibly straight → cannot be Sujud (must be Ruku)
    #      This is safe: from the front during Ruku knee_angle reads
    #      ~163 (visible). During real Sujud it reads 90 (occluded)
    #      or below 110 (kneeling) — both pass the < 140 check.

    nose_low           = nose_y > 0.70
    nose_below_hip     = nose_y > hip_y + 0.05
    shoulder_below_hip = a["shoulder_y"] > a["hip_y"] + 0.10
    hips_raised        = hip_y < 0.75
    elbow_bent         = a["elbow_angle"] <= SUJUD_ELBOW_MAX
    elbow_occluded     = a["elbow_angle"] >= 85.0
    knees_not_straight = knee < 140

    # Primary A — nose-based
    primary_A = nose_low and hips_raised and knees_not_straight
    if primary_A and elbow_bent:
        return "SUJUD"
    if primary_A and elbow_occluded:
        return "SUJUD"

    # Primary B — shoulder-based (face hidden under telekung)
    primary_B = shoulder_below_hip and hips_raised and knees_not_straight
    if primary_B and elbow_bent:
        return "SUJUD"
    if primary_B and elbow_occluded:
        return "SUJUD"

    # Tertiary — nose below hip
    if nose_below_hip and hips_raised and knees_not_straight and elbow_bent:
        return "SUJUD"
    if nose_below_hip and hips_raised and knees_not_straight and elbow_occluded:
        return "SUJUD"
    if nose_below_hip and hips_raised and elbow_occluded:
        return "SUJUD"

    # 2. TASHAHHUD — sitting (knee and hip both bent)
    #
    # Front-view hip angle issue: from the front, when sitting the knee
    # landmark overlaps with the hip in 2D projection, making hip_angle
    # read falsely large (up to ~165°) even during a genuine sitting posture.
    # TASHAHHUD_HIP_MAX = 155 (config) is calibrated for side view where
    # the angle is reliable. For front view we raise the ceiling to 170°
    # to accommodate this projection compression.
    #
    # Qiyam safety guard — knee_angle > 140 means knees are visibly straight
    # → person is standing, not sitting. During Tashahhud knees are bent so
    # knee_angle reads low or returns 90.0 (neutral fallback when ankles
    # occluded under telekung). Both cases pass the <= 140 check.
    # This prevents a standing Qiyam from accidentally matching if hip_angle
    # happens to read below 170 due to landmark drift.
    TASHAHHUD_HIP_MAX_FRONT = 170   # relaxed from 155 for front-view projection
    if (knee <= TASHAHHUD_KNEE_MAX
            and hip <= TASHAHHUD_HIP_MAX_FRONT
            and spine < 50
            and knee <= 140):       # knees not straight — confirms sitting not standing
        return "TASHAHHUD"

    # 3. RUKU — from front view
    #
    # The original spine > 20° guard was added to prevent Qiyam false positives
    # when camera distance compresses sh_hip_y_diff. However, when bowing
    # TOWARD the camera (direct front view), the shoulders move in Z (depth)
    # not X, so the 2D spine_angle stays near 0° even during a full bow.
    # The spine guard therefore incorrectly blocks valid Ruku in front-view.
    #
    # Replacement strategy — use hip_angle and ear position instead:
    #
    # Path A — strong signal: both Y-proximity conditions pass
    #   sh_hip_y_diff < 0.15  : shoulders descended toward hip level
    #   ear_sh_y_diff < 0.12  : head bowing forward to shoulder level
    #   hip_angle > 140       : hips/knees still relatively straight
    #                           (rules out Tashahhud which has hip ~100°)
    #   This fires even when spine_angle ≈ 0 (direct front-view camera).
    #
    # Path B — looser signal: only shoulder drop visible (head covered)
    #   sh_hip_y_diff < 0.12  : tighter threshold since head signal absent
    #   hip_angle > 140       : straight hips confirm not sitting
    #   spine < 60            : some forward lean — blocks pure upright
    #   knee > 130            : knees not bent (not Tashahhud/Sujud)
    #
    # Qiyam guard: if sh_hip_y_diff is small BUT sh_hip_y_diff >= 0.09
    # AND ear is still well above shoulders (ear_sh_y_diff > 0.15),
    # it is likely Qiyam at close range — do not classify as Ruku.

    # Path A — head bowing clearly visible
    ruku_A = (sh_hip_y_diff < 0.15
              and ear_sh_y_diff < 0.12
              and hip > 140)
    if ruku_A:
        return "RUKU"

    # Path B — head hidden (telekung), rely on shoulder drop + straight legs
    ruku_B = (sh_hip_y_diff < 0.12
              and hip > 140
              and spine < 60
              and knee > 130)
    if ruku_B:
        return "RUKU"

    # 4. QIYAM — shoulders well above hips, spine upright
    if sh_hip_y_diff >= 0.18 and spine < 35:
        return "QIYAM"
    # Fallback: upright spine with reasonable shoulder-hip separation
    if spine < 25 and sh_hip_y_diff >= 0.12:
        return "QIYAM"
    # Last resort upright fallback
    # Guard: sh_hip_y_diff >= 0.15 required — when sitting (Tashahhud) the
    # shoulders are closer to the hips in Y because the body is compact.
    # Without this guard, spine < 15 fires during Tashahhud because sitting
    # upright also produces a near-vertical spine reading from the front.
    if spine < 15 and sh_hip_y_diff >= 0.15:
        return "QIYAM"

    return "UNKNOWN"


# ── View-fused classifier ────────────────────────────────────

def classify_posture(lm):
    """
    Classify prayer posture using view-aware fusion.

    If view is detected as 'side'  → use side-view classifier
    If view is detected as 'front' → use front-view classifier
    If view is 'unknown'           → run both and take the non-UNKNOWN result;
                                     if both agree, return that; else UNKNOWN.

    Returns: 'QIYAM' | 'RUKU' | 'SUJUD' | 'TASHAHHUD' | 'UNKNOWN'
    """
    a    = get_joint_angles(lm)
    view = a["view"]

    if view == "side":
        return _classify_side(a)

    if view == "front":
        return _classify_front(a)

    # Unknown view — run both and fuse
    side_result  = _classify_side(a)
    front_result = _classify_front(a)

    if side_result == front_result:
        return side_result                   # both agree
    if side_result != "UNKNOWN":
        return side_result                   # side is usually more reliable
    if front_result != "UNKNOWN":
        return front_result
    return "UNKNOWN"


# ── Transition smoother ───────────────────────────────────────

class TransitionSmoother:
    """
    Implements professor directive #2:
    "Smoothing specifically during posture transitions."

    Standard majority-vote smoothing treats all frames equally.
    This smoother detects when a posture change is in progress and
    applies a stricter confirmation requirement during that window,
    preventing brief intermediate postures from being output.

    Behaviour:
      - STABLE state: output label as soon as simple majority holds
      - TRANSITIONING state: require TRANSITION_SMOOTH_FRAMES consecutive
        frames of the new label before committing to it
    """

    STATE_STABLE       = "stable"
    STATE_TRANSITIONING = "transitioning"

    def __init__(self):
        self._window         = max(SMOOTHING_WINDOW, 7)
        self._buffer         = deque(maxlen=self._window)
        self._confirmed      = None   # last fully confirmed posture
        self._state          = self.STATE_STABLE
        self._candidate      = None   # posture we are considering switching to
        self._candidate_count = 0     # consecutive frames of candidate seen

    def update(self, raw):
        """
        Accept a raw label (already through FramePreprocessor).
        Returns the smoothed label to pass to the FSM.
        """
        self._buffer.append(raw)

        # ── STABLE state ─────────────────────────────────────
        if self._state == self.STATE_STABLE:
            majority = self._majority()

            # No change — keep confirmed
            if majority == self._confirmed or majority in ("UNKNOWN", "TRANSITION"):
                return self._confirmed or majority

            # A new posture is appearing — enter transition mode
            self._state          = self.STATE_TRANSITIONING
            self._candidate      = majority
            self._candidate_count = 1
            # During transition, output TRANSITION so the FSM waits
            return "TRANSITION"

        # ── TRANSITIONING state ──────────────────────────────
        if raw == self._candidate:
            self._candidate_count += 1
        elif raw not in ("UNKNOWN", "TRANSITION"):
            # The candidate changed — a different posture appeared
            # mid-transition; reset candidate
            self._candidate       = raw
            self._candidate_count = 1

        # Enough consecutive frames of candidate — commit
        if self._candidate_count >= TRANSITION_SMOOTH_FRAMES:
            self._confirmed = self._candidate
            self._state     = self.STATE_STABLE
            self._candidate = None
            self._candidate_count = 0
            return self._confirmed

        # Still in transition
        return "TRANSITION"

    def _majority(self):
        if not self._buffer:
            return "UNKNOWN"
        return max(set(self._buffer), key=self._buffer.count)

    def reset(self):
        self._buffer.clear()
        self._confirmed       = None
        self._state           = self.STATE_STABLE
        self._candidate       = None
        self._candidate_count = 0


# ── SmoothedClassifier (public interface unchanged) ───────────

class SmoothedClassifier:
    """
    Public interface used by main.py.
    Combines:
      1. Raw classify_posture (view-fused, telekung-robust)
      2. TransitionSmoother (direction #2)

    The FramePreprocessor (direction #3) is applied separately
    in main.py after this class returns its label.
    """

    def __init__(self):
        self._smoother = TransitionSmoother()

    def update(self, lm):
        raw = classify_posture(lm)
        return self._smoother.update(raw)

    def get_debug_angles(self, lm):
        return get_joint_angles(lm)

    def reset(self):
        self._smoother.reset()