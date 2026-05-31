# conftest.py — pytest configuration for SalatVision
#
# Stubs out mediapipe.solutions when the full MediaPipe install is
# unavailable (e.g. CI, sandboxed environments, machines without
# the GPU/platform dependencies MediaPipe requires).
#
# On your development machine with a working MediaPipe install this
# file does nothing — the real library is used automatically.
#
# How it works:
#   conftest.py is loaded by pytest before any test module imports.
#   We check whether mp.solutions is accessible and, if not, inject
#   a lightweight stub that satisfies the imports in posture_classifier
#   and frame_preprocessor without actually running pose estimation.

import sys
import types


def _mediapipe_is_broken():
    try:
        import mediapipe as mp
        _ = mp.solutions.pose   # the access that fails in broken installs
        return False
    except (AttributeError, ImportError):
        return True


if _mediapipe_is_broken():
    # ── Build a minimal mediapipe stub ──────────────────────────
    # Only the symbols that posture_classifier.py and frame_preprocessor.py
    # actually access at import time are needed here.

    mp_stub = types.ModuleType("mediapipe")

    # mediapipe.solutions.pose.PoseLandmark
    # PoseLandmark is an enum whose .value gives the landmark index (0-32).
    # We replicate only the landmarks referenced in the source files.
    class _FakeLandmark:
        def __init__(self, value):
            self.value = value

    class _PoseLandmark:
        NOSE           = _FakeLandmark(0)
        LEFT_EYE_INNER = _FakeLandmark(1)
        LEFT_EYE       = _FakeLandmark(2)
        RIGHT_EYE_INNER= _FakeLandmark(3)
        RIGHT_EYE      = _FakeLandmark(4)
        LEFT_EAR       = _FakeLandmark(7)
        RIGHT_EAR      = _FakeLandmark(8)
        LEFT_SHOULDER  = _FakeLandmark(11)
        RIGHT_SHOULDER = _FakeLandmark(12)
        LEFT_ELBOW     = _FakeLandmark(13)
        RIGHT_ELBOW    = _FakeLandmark(14)
        LEFT_WRIST     = _FakeLandmark(15)
        RIGHT_WRIST    = _FakeLandmark(16)
        LEFT_HIP       = _FakeLandmark(23)
        RIGHT_HIP      = _FakeLandmark(24)
        LEFT_KNEE      = _FakeLandmark(25)
        RIGHT_KNEE     = _FakeLandmark(26)
        LEFT_ANKLE     = _FakeLandmark(27)
        RIGHT_ANKLE    = _FakeLandmark(28)

    class _FakePose:
        """Stub for mp.solutions.pose.Pose — never called in unit tests."""
        POSE_CONNECTIONS = []

        def __init__(self, **kwargs):
            pass
        def process(self, image):
            return None
        def close(self):
            pass

    class _PoseSolutions:
        pose           = types.SimpleNamespace(
            Pose           = _FakePose,
            PoseLandmark   = _PoseLandmark,
            POSE_CONNECTIONS = [],
        )
        drawing_utils  = types.SimpleNamespace(
            draw_landmarks = lambda *a, **k: None,
            DrawingSpec    = lambda **k: None,
        )

    mp_stub.solutions = _PoseSolutions()

    # Register so that `import mediapipe as mp` returns our stub
    sys.modules["mediapipe"] = mp_stub
    print("\n[conftest] MediaPipe stub injected (mp.solutions unavailable in this environment)")