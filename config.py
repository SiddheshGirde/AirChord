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
NUM_HANDS = 1                          # AirChord only needs to track one hand
MIN_HAND_DETECTION_CONFIDENCE = 0.6
MIN_HAND_PRESENCE_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.6
