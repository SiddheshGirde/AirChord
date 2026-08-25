"""
config.py
----------
Central place for AirChord's configuration constants.
Keeping settings here means you can tweak the camera index,
resolution, or window name without touching the app logic.
"""

import os

# --- Camera settings ---
CAMERA_INDEX = 0          # Default webcam. Try 1, 2... if you have multiple cameras.
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# --- Window / UI settings ---
WINDOW_NAME = "AirChord"
QUIT_KEY = "q"             # Key to press (inside the video window) to quit

# --- Robustness settings ---
MAX_CONSECUTIVE_FRAME_FAILURES = 30   # Give up after this many failed reads in a row

# --- Hand detection settings (Phase 2) ---
MODEL_PATH = os.path.join("models", "hand_landmarker.task")
NUM_HANDS = 2                          # Phase 3: track both hands at once
MIN_HAND_DETECTION_CONFIDENCE = 0.6
MIN_HAND_PRESENCE_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.6

# --- Two-hand roles (Phase 3) ---
CHORD_HAND = "Left"        # This hand plays the fixed chord gestures
DYNAMICS_HAND = "Right"    # This hand controls continuous dynamics/expression

# --- Gesture recognition tuning (Phase 3) ---
# Thumb tip must be at least this many "palm-widths" from the pinky-base
# landmark to count as extended. Handedness-independent by design.
THUMB_EXTENDED_RATIO = 1.0
# calculate_dynamics() normalizes avg-fingertip-spread / palm-width between
# these two values. These are reasonable starting points - nudge them while
# watching the on-screen dynamics bar if it feels off on your hand/camera.
DYNAMICS_MIN_SPREAD_RATIO = 1.2   # ~closed hand -> dynamics 0.0
DYNAMICS_MAX_SPREAD_RATIO = 2.4   # ~fully open hand -> dynamics 1.0
GESTURE_STABLE_FRAMES = 5         # consecutive frames a gesture must hold to be "confirmed"

# --- Scale settings (Phase 3) ---
DEFAULT_ROOT_NOTE = "C"

# --- Audio synthesis settings (Phase 4) ---
CHORD_DURATION_SECONDS = 1.2   # length of one synthesized chord "strum"
CHORD_DECAY_RATE = 1.5         # higher = faster fade-out (exponential decay)
