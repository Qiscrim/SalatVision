# ============================================================
# frame_preprocessor.py — Pre-processing pipeline
#
# Implements professor directive #3:
#   "Add a pre-processing stage to remove noisy frames and
#    use interpolation for transitions."
#
# What it does:
#   1. NOISE DETECTION — flags a frame as noisy if:
#        • Too few landmarks are visible (< MIN_VISIBLE_LANDMARKS)
#        • The overall landmark confidence is below threshold
#        • The detected posture jumps erratically (not a smooth transition)
#
#   2. INTERPOLATION — when a noisy frame is removed, the gap is
#      filled by carrying the last known good posture forward until
#      a new stable posture is confirmed.
#
#   3. TRANSITION LABELLING — frames between two different confirmed
#      postures are labelled "TRANSITION" instead of UNKNOWN, so the
#      FSM can distinguish "still moving" from "genuinely undetected".
# ============================================================

from collections import deque
from config import (
    SMOOTHING_WINDOW,
    VISIBILITY_THRESHOLD,
    MIN_VISIBLE_LANDMARKS,
    TRANSITION_WINDOW,
    NOISE_JUMP_THRESHOLD,
)
import mediapipe as mp

mp_pose = mp.solutions.pose
PL      = mp_pose.PoseLandmark

# Key landmarks we care about — if fewer than MIN_VISIBLE_LANDMARKS
# of these are visible, the frame is considered noisy.
KEY_LANDMARKS = [
    PL.NOSE.value,
    PL.LEFT_SHOULDER.value,  PL.RIGHT_SHOULDER.value,
    PL.LEFT_HIP.value,       PL.RIGHT_HIP.value,
    PL.LEFT_KNEE.value,      PL.RIGHT_KNEE.value,
    PL.LEFT_WRIST.value,     PL.RIGHT_WRIST.value,
]

VALID_POSTURES = {"QIYAM", "RUKU", "SUJUD", "TASHAHHUD"}

# How many frames of shoulder_y history to track for trajectory prediction
TRAJECTORY_WINDOW = 6


class FramePreprocessor:
    """
    Sits between the raw classifier output and the FSM.

    Call process(raw_label, landmarks) every frame.
    Returns a cleaned label: one of the four postures, "TRANSITION",
    or "UNKNOWN".

    Added: Sujud trajectory prediction.
    When transitioning from QIYAM toward Sujud, the body passes through
    a stage where landmarks are partially visible but the classifier
    returns UNKNOWN or TRANSITION. By tracking the rate of change of
    shoulder_y (how fast the shoulders are descending in the frame),
    the preprocessor can predict Sujud is imminent and return SUJUD
    early — before the classifier fully confirms it.

    Conditions for early Sujud prediction:
      1. Last confirmed posture was QIYAM (coming from standing)
      2. shoulder_y has been consistently rising (person bending down)
         over the last TRAJECTORY_WINDOW frames
      3. Current shoulder_y is already low enough (>= 0.60)
      4. Current hip_y is below 0.75 (body is already partway down)
      5. Raw label is UNKNOWN or TRANSITION (classifier not yet sure)
    """

    def __init__(self):
        self._raw_history   = deque(maxlen=NOISE_JUMP_THRESHOLD + 2)
        self._last_good     = None
        self._noisy_streak  = 0
        self._clean_history = deque(maxlen=TRANSITION_WINDOW)
        # Trajectory prediction: track shoulder_y over last N frames
        self._shoulder_y_history = deque(maxlen=TRAJECTORY_WINDOW)

    def process(self, raw_label, landmarks):
        """
        Parameters
        ----------
        raw_label : str
            Output from SmoothedClassifier.update()
        landmarks : list
            MediaPipe landmark list (33 objects with .visibility)

        Returns
        -------
        str  — cleaned label passed to the FSM
        """
        # Track shoulder_y every frame for trajectory prediction
        if landmarks is not None:
            try:
                l_sh = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                r_sh = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                shoulder_y = (l_sh.y + r_sh.y) / 2
                self._shoulder_y_history.append(shoulder_y)

                l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                hip_y = (l_hip.y + r_hip.y) / 2
            except Exception:
                shoulder_y = None
                hip_y = None
        else:
            shoulder_y = None
            hip_y = None

        # Step 1: noise detection
        noisy = self._is_noisy_frame(landmarks, raw_label)

        if noisy:
            self._noisy_streak += 1
            if self._last_good and self._noisy_streak <= 8:
                return self._last_good
            else:
                return "UNKNOWN"

        # Step 2: clean frame — reset streak
        self._noisy_streak = 0
        self._raw_history.append(raw_label)

        # Step 3: transition label resolution
        clean_label = self._resolve_transition(raw_label)

        # Step 4: Sujud trajectory prediction
        # If the classifier is uncertain (UNKNOWN/TRANSITION) but the
        # shoulder is consistently descending from a QIYAM position,
        # predict SUJUD early rather than waiting for full confirmation.
        if clean_label in ("UNKNOWN", "TRANSITION"):
            predicted = self._predict_sujud(shoulder_y, hip_y)
            if predicted:
                clean_label = "SUJUD"

        # Step 5: update last good
        if clean_label in VALID_POSTURES:
            self._last_good = clean_label

        self._clean_history.append(clean_label)
        return clean_label

    def reset(self):
        self._raw_history.clear()
        self._last_good    = None
        self._noisy_streak = 0
        self._clean_history.clear()
        self._shoulder_y_history.clear()

    # ── Private ──────────────────────────────────────────────

    def _is_noisy_frame(self, landmarks, raw_label):
        """
        A frame is noisy if:
          (a) Too few key landmarks are visible — body probably not in frame
          (b) The label is UNKNOWN AND there was a valid posture recently
              AND the history is too erratic (rapid label jumping)
        """
        if landmarks is None:
            return True

        # Count how many key landmarks pass the visibility threshold
        visible = sum(
            1 for idx in KEY_LANDMARKS
            if _vis(landmarks, idx) >= VISIBILITY_THRESHOLD
        )
        if visible < MIN_VISIBLE_LANDMARKS:
            return True

        # Erratic jump detection: if the last N raw labels contain
        # 3+ different posture labels, the frame sequence is unstable
        if len(self._raw_history) >= NOISE_JUMP_THRESHOLD:
            recent_set = set(self._raw_history) - {"UNKNOWN", "TRANSITION"}
            if len(recent_set) >= 3:
                return True

        return False

    def _predict_sujud(self, shoulder_y, hip_y):
        """
        Predict Sujud early based on shoulder descent trajectory.

        Covers two transitions:
          QIYAM → SUJUD: person bends from standing into prostration
          TASHAHHUD → SUJUD: person folds forward from sitting

        Conditions:
          1. Last confirmed posture was QIYAM or TASHAHHUD
          2. shoulder_y consistently rising (shoulders descending in frame)
             across last TRAJECTORY_WINDOW frames
          3. shoulder_y already >= threshold (shoulders low enough)
          4. hip_y < 0.75 (hips raised — body partway into prostration)

        TASHAHHUD → SUJUD uses a slightly lower shoulder threshold (0.55)
        because from sitting, the shoulders start lower in the frame
        compared to standing, so the descent begins from a lower baseline.
        """
        if self._last_good not in ("QIYAM", "TASHAHHUD"):
            return False
        if shoulder_y is None or hip_y is None:
            return False
        if len(self._shoulder_y_history) < TRAJECTORY_WINDOW:
            return False

        # Check shoulder_y has been consistently rising (moving down in frame)
        history = list(self._shoulder_y_history)
        descending_count = 0
        for i in range(1, len(history)):
            if history[i] > history[i-1]:
                descending_count += 1

        # Require majority of frames show descent
        if descending_count < TRAJECTORY_WINDOW // 2:
            return False

        # Shoulder threshold — slightly different per source posture
        # From QIYAM: shoulders start high, need to descend significantly
        # From TASHAHHUD: already lower baseline, less descent needed
        shoulder_threshold = 0.60 if self._last_good == "QIYAM" else 0.55
        if shoulder_y < shoulder_threshold:
            return False

        # Hip must be raised — body is prostrating, not just leaning
        if hip_y >= 0.75:
            return False

        return True

    def _resolve_transition(self, raw_label):
        """
        If the raw label is UNKNOWN but we have a recent clean history
        that was moving from one posture toward another, label it
        TRANSITION instead of UNKNOWN.

        A transition is detected when:
          - The current label is UNKNOWN
          - The last_good posture was a valid posture
          - We haven't been in a stable new posture yet
        """
        if raw_label in VALID_POSTURES:
            return raw_label

        # raw_label is UNKNOWN — check if we are mid-transition
        if self._last_good in VALID_POSTURES:
            return "TRANSITION"

        return "UNKNOWN"


def _vis(lm, idx):
    """Safe visibility accessor."""
    try:
        return lm[idx].visibility
    except (AttributeError, IndexError):
        return 0.0