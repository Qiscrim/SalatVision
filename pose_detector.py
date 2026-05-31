# ============================================================
# pose_detector.py — MediaPipe Pose wrapper
#
# Updated: Added LandmarkSmoother implementing professor directive:
#   "Apply smoothing to each joint's x and y coordinate values
#    before angle calculation — Gaussian, averaging, or median filtering"
#
# Method chosen: MEDIAN FILTERING
#   A rolling buffer of the last N frames is kept for each landmark.
#   The median of the buffer is returned instead of the raw value.
#   Median completely ignores sudden spikes (e.g. x jumps from 20 to 100
#   due to fabric occlusion) — averaging and Gaussian are both pulled
#   toward the spike, but median is not.
#
# Where it sits in the pipeline:
#   Camera → PoseDetector (raw landmarks → LandmarkSmoother → smoothed)
#          → posture_classifier → frame_preprocessor → FSM
# ============================================================

import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from config import (
    MEDIAPIPE_DETECTION_CONFIDENCE,
    MEDIAPIPE_TRACKING_CONFIDENCE,
    MEDIAPIPE_MODEL_COMPLEXITY,
    LANDMARK_SMOOTH_WINDOW,
)


# ── Landmark Smoother ─────────────────────────────────────────

class LandmarkSmoother:
    """
    Applies median filtering to each landmark's x and y coordinates
    independently across the last LANDMARK_SMOOTH_WINDOW frames.

    Why median and not averaging or Gaussian:
      - Averaging: a spike value (e.g. x=100 when normally x=20) pulls
        the average away from the true value
      - Gaussian: assigns weighted average — recent frames get more weight,
        but the spike still contributes to the result
      - Median: returns the middle value of the sorted buffer. A single
        spike is completely ignored as long as the buffer has more than
        one other value. Most robust for single-frame outliers caused by
        fabric occlusion or detection error.

    MediaPipe already applies internal Gaussian smoothing via
    smooth_landmarks=True. This median filter is a second layer
    specifically targeting sudden spikes that pass through MediaPipe's
    internal filter.
    """

    def __init__(self, n_landmarks=33, window=None):
        if window is None:
            window = LANDMARK_SMOOTH_WINDOW
        self._window = window
        self._n      = n_landmarks

        # One deque per landmark per coordinate (x, y)
        # Shape: [n_landmarks] → {x: deque, y: deque}
        self._buffers = [
            {
                "x": deque(maxlen=window),
                "y": deque(maxlen=window),
            }
            for _ in range(n_landmarks)
        ]

    def smooth(self, landmarks):
        """
        Accept a raw MediaPipe landmark list (33 objects with .x, .y, .visibility).
        Return a list of SmoothedLandmark objects with median-filtered x and y,
        and unchanged visibility and z.

        Parameters
        ----------
        landmarks : list
            results.pose_landmarks.landmark from MediaPipe

        Returns
        -------
        list of SmoothedLandmark
        """
        smoothed = []
        for i, lm in enumerate(landmarks):
            # Add raw values to buffer
            self._buffers[i]["x"].append(lm.x)
            self._buffers[i]["y"].append(lm.y)

            # Compute median of buffer
            # np.median handles deques correctly
            sx = float(np.median(self._buffers[i]["x"]))
            sy = float(np.median(self._buffers[i]["y"]))

            smoothed.append(SmoothedLandmark(
                x          = sx,
                y          = sy,
                z          = lm.z,           # z not used — pass through unchanged
                visibility = lm.visibility,  # visibility unchanged — raw confidence
            ))

        return smoothed

    def reset(self):
        """Clear all buffers — call when prayer resets."""
        for buf in self._buffers:
            buf["x"].clear()
            buf["y"].clear()


class SmoothedLandmark:
    """
    Lightweight container matching MediaPipe's landmark interface.
    Allows the rest of the code (posture_classifier, frame_preprocessor)
    to use smoothed landmarks without any changes.
    """
    __slots__ = ("x", "y", "z", "visibility")

    def __init__(self, x, y, z, visibility):
        self.x          = x
        self.y          = y
        self.z          = z
        self.visibility = visibility


# ── Pose Detector ─────────────────────────────────────────────

class PoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose    = self.mp_pose.Pose(
            min_detection_confidence = MEDIAPIPE_DETECTION_CONFIDENCE,
            min_tracking_confidence  = MEDIAPIPE_TRACKING_CONFIDENCE,
            model_complexity         = MEDIAPIPE_MODEL_COMPLEXITY,
            smooth_landmarks         = True,   # MediaPipe internal Gaussian smoothing
                                               # (1st layer — handles gradual drift)
        )
        # 2nd layer — median filter for sudden spikes
        self._smoother = LandmarkSmoother()

    def process(self, frame):
        """
        Process a camera frame and return smoothed landmarks.

        Returns: (smoothed_landmarks | None, results)

        Pipeline:
          raw frame
            → MediaPipe Pose (internal Gaussian smooth via smooth_landmarks=True)
            → LandmarkSmoother (median filter per joint x,y)
            → returned to classifier
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.pose.process(rgb)
        rgb.flags.writeable = True

        if results.pose_landmarks:
            raw      = results.pose_landmarks.landmark
            smoothed = self._smoother.smooth(raw)   # apply median filter
            return smoothed, results

        return None, results

    def reset_smoother(self):
        """Call when prayer resets to clear the coordinate buffer."""
        self._smoother.reset()

    def draw_skeleton(self, frame, results):
        if results and results.pose_landmarks:
            self.mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_draw.DrawingSpec(
                    color=(80, 200, 120), thickness=2, circle_radius=3
                ),
                connection_drawing_spec=self.mp_draw.DrawingSpec(
                    color=(60, 140, 200), thickness=2
                ),
            )
        return frame

    def release(self):
        self.pose.close()