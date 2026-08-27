"""
hand_detector.py
-----------------
Wraps MediaPipe's HandLandmarker task (the current MediaPipe Tasks API -
the older `mp.solutions.hands` API was retired in 2023 in favor of this one).

Responsibilities:
    - Download the hand_landmarker.task model bundle on first run
    - Run hand detection + 21-landmark extraction on a video frame
    - Draw the landmarks, skeleton connections, and Left/Right label
      onto the frame for visualization
"""

import os
import time
import urllib.request

import cv2
import mediapipe as mp

import config

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MIN_VALID_MODEL_SIZE_BYTES = 1_000_000  # ~1 MB - guards against a partial/corrupt download

# The 21 MediaPipe hand landmarks, connected as a "skeleton" for drawing.
# (0=wrist, 1-4=thumb, 5-8=index, 9-12=middle, 13-16=ring, 17-20=pinky)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12),     # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),   # Ring finger
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky + palm
]

# Fingertip landmarks get their own color (inspired by gesture-synth style
# visualizations, where each finger reads as its own "control"); every
# other joint is plain white.
FINGERTIP_COLORS = {
    4: (0, 165, 255),     # Thumb tip - orange
    8: (0, 255, 255),     # Index tip - yellow
    12: (255, 0, 255),    # Middle tip - magenta
    16: (255, 255, 0),    # Ring tip - cyan
    20: (203, 192, 255),  # Pinky tip - pink
}
JOINT_COLOR = (255, 255, 255)
CONNECTION_COLOR = (0, 200, 0)


def ensure_model_downloaded(model_path: str):
    """Download the hand_landmarker.task model bundle if it's missing or looks corrupt."""
    if os.path.exists(model_path) and os.path.getsize(model_path) > MIN_VALID_MODEL_SIZE_BYTES:
        return

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    print(f"[INFO] Downloading hand landmark model to '{model_path}' (first run only, ~8 MB)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, model_path)
        print("[INFO] Model downloaded successfully.")
    except Exception as e:
        raise RuntimeError(
            "Could not download the hand landmark model automatically "
            f"({e}).\nDownload it manually from:\n  {MODEL_URL}\n"
            f"and save it as:\n  {model_path}"
        )


class HandDetector:
    """Detects hands in video frames and extracts 21 landmarks per hand."""

    def __init__(self, model_path: str = config.MODEL_PATH, num_hands: int = config.NUM_HANDS):
        ensure_model_downloaded(model_path)

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=config.MIN_HAND_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=config.MIN_HAND_PRESENCE_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )
        try:
            self._landmarker = HandLandmarker.create_from_options(options)
        except Exception as e:
            raise RuntimeError(
                f"Could not load the hand landmark model at '{model_path}' ({e}). "
                "It may be corrupted - try deleting the models/ folder and "
                "running again so it re-downloads."
            )
        self._start_time = time.time()

    def detect(self, frame_bgr):
        """
        Run hand detection on one BGR frame (as read by OpenCV).

        Returns a HandLandmarkerResult. result.hand_landmarks is a list of
        detected hands; each hand is a list of 21 landmarks with .x, .y, .z
        normalized to [0, 1] relative to frame width/height. result.handedness
        gives Left/Right + confidence for each detected hand.
        """
        # MediaPipe expects RGB; OpenCV gives BGR - convert before wrapping.
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        timestamp_ms = int((time.time() - self._start_time) * 1000)
        return self._landmarker.detect_for_video(mp_image, timestamp_ms)

    def draw_landmarks(self, frame, result):
        """Draw the 21 landmarks + hand skeleton + Left/Right label onto the frame."""
        if not result.hand_landmarks:
            return frame

        h, w = frame.shape[:2]

        for hand_index, hand_landmarks in enumerate(result.hand_landmarks):
            points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

            for start_idx, end_idx in HAND_CONNECTIONS:
                cv2.line(frame, points[start_idx], points[end_idx], CONNECTION_COLOR, 2)

            for idx, point in enumerate(points):
                color = FINGERTIP_COLORS.get(idx, JOINT_COLOR)
                radius = 7 if idx in FINGERTIP_COLORS else 4
                cv2.circle(frame, point, radius, color, -1)

            if result.handedness and hand_index < len(result.handedness):
                label = result.handedness[hand_index][0].category_name
                wrist = points[0]
                cv2.putText(frame, label, (wrist[0] - 20, wrist[1] + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame

    def close(self):
        """Release the MediaPipe landmarker's resources."""
        self._landmarker.close()
