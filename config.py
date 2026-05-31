# ============================================================
# config.py — Salat Tracker Configuration
# Updated for professor directives:
#   #1 Front view support (view detection thresholds)
#   #2 Transition smoothing (TRANSITION_SMOOTH_FRAMES)
#   #3 Pre-processing (VISIBILITY_THRESHOLD, MIN_VISIBLE_LANDMARKS,
#       TRANSITION_WINDOW, NOISE_JUMP_THRESHOLD)
# ============================================================

# ── Posture Angle Thresholds (degrees) ──────────────────────
# Tuned for telekung/hijab wearers

# Qiyam (Standing)
QIYAM_HIP_MIN = 145        # relaxed from 155 — telekung bulk compresses reading

# Ruku (Bowing)
RUKU_HIP_MAX  = 140        # relaxed from 130
RUKU_HIP_MIN  = 45         # relaxed from 50

# Sujud (Prostration)
# Legacy value — now detected via nose Y position in posture_classifier.py
SUJUD_SHOULDER_HIP_DIFF = 0.05

# Elbow angle threshold for Sujud confirmation (degrees).
# In prostration, elbows are bent as arms support body weight on the ground.
# Mughal's reference uses < 100°; we use 110° to be more tolerant for
# telekung wearers whose sleeves may slightly shift the elbow landmark.
# If elbow landmarks are occluded (returns 90.0 neutral fallback),
# the classifier accepts Sujud on nose + knee alone.
SUJUD_ELBOW_MAX = 110

# Tashahhud (Sitting)
TASHAHHUD_KNEE_MAX = 120   # relaxed from 115
TASHAHHUD_HIP_MAX  = 155

# ── State Machine ────────────────────────────────────────────
POSTURE_HOLD_TIME = 0.5    # seconds — raised from 0.6 for telekung noise

# ── Smoothing ────────────────────────────────────────────────
SMOOTHING_WINDOW = 7       # majority-vote buffer size — raised from 5

# Directive #2 — Transition smoothing
TRANSITION_SMOOTH_FRAMES = 3   # ~120ms at 25fps, originally 5 frames — reduced to 3 for more responsive transitions

# Directive #4 — Joint coordinate smoothing (professor directive)
# Number of frames kept in the per-joint median filter buffer.
# Each landmark's x and y are smoothed independently across this window.
# 5 frames ≈ ~120ms at 25fps — enough to absorb single-frame spikes
# without introducing noticeable lag in the angle readings.
LANDMARK_SMOOTH_WINDOW = 5

# ── Pre-processing (Directive #3) ────────────────────────────

# Minimum landmark visibility score to trust a joint (0.0–1.0)
VISIBILITY_THRESHOLD = 0.45

# Minimum number of key landmarks that must be visible
# for a frame to be considered usable (out of 9 key landmarks)
MIN_VISIBLE_LANDMARKS = 4

# Size of the window used to detect transition state
TRANSITION_WINDOW = 10    # frames

# If the last N raw labels contain 3+ different postures,
# the frame sequence is flagged as erratically noisy
NOISE_JUMP_THRESHOLD = 6  # frames

# ── Prayer Types ─────────────────────────────────────────────
PRAYER_TYPES = {
    "Fajr":    2,
    "Maghrib": 3,
    "Zuhr":    4,
    "Asr":     4,
    "Isha":    4,
    "Custom":  99,
}

# ── Rakaat Sequence ──────────────────────────────────────────
RAKAAT_SEQUENCE = ["QIYAM", "RUKU", "QIYAM", "SUJUD", "TASHAHHUD", "SUJUD"]

# ── UI Settings ──────────────────────────────────────────────
WINDOW_NAME  = "Salat Tracker"
CAMERA_INDEX = 0
FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720

# Colors (BGR)
COLOR_GREEN  = (0, 220, 120)
COLOR_GOLD   = (0, 200, 255)
COLOR_WHITE  = (255, 255, 255)
COLOR_DARK   = (20, 20, 30)
COLOR_GRAY   = (160, 160, 160)
COLOR_RED    = (60, 60, 220)
COLOR_TEAL   = (200, 210, 60)

# Posture display colors (BGR)
POSTURE_COLORS = {
    "QIYAM":      (0, 220, 120),
    "RUKU":       (0, 200, 255),
    "SUJUD":      (80, 160, 255),
    "TASHAHHUD":  (200, 120, 255),
    "TRANSITION": (0, 180, 255),   # amber — moving between postures
    "UNKNOWN":    (100, 100, 100),
}

# ── MediaPipe ────────────────────────────────────────────────
MEDIAPIPE_DETECTION_CONFIDENCE = 0.65 # from 0.65 to 0.5 for better detection with telekung
MEDIAPIPE_TRACKING_CONFIDENCE  = 0.5 # from 0.60 to 0.5 to maintain tracking with telekung occlusions
MEDIAPIPE_MODEL_COMPLEXITY     = 1 # 0=light, 1=full, 2=heavy — we use heavy for better accuracy at the cost of CPU
   
# ── Audio Settings ───────────────────────────────────────────
AUDIO_ENABLED        = True
AUDIO_RATE           = 130
AUDIO_VOLUME         = 0.85
AUDIO_ANNOUNCE_START = True
AUDIO_ANNOUNCE_END   = True