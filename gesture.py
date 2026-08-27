"""
gesture.py
-----------
Turns raw MediaPipe hand-landmark detections into AirChord-specific meaning:
    - Splits a two-hand detection into "chord hand" and "dynamics hand"
      (which physical hand plays which role is set in config.py)
    - Classifies the chord hand's finger-state pattern into a named gesture
      by counting extended fingers (1-4), and smooths it over a few frames
      so it doesn't flicker
    - Maps a gesture to a chord name
    - Turns the dynamics hand's VERTICAL POSITION into a continuous
      0.0-1.0 value - raise your hand for louder, matching the convention
      most hand-tracking gesture-synth apps use

This is pure geometry on landmark coordinates - no ML model, no audio -
matching the "don't rely on image classification" requirement.
"""

import config

# --- Landmark index constants (MediaPipe's 21-point hand topology) ---
WRIST = 0
THUMB_TIP, THUMB_IP = 4, 3
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_PIP, MIDDLE_TIP = 10, 12
RING_PIP, RING_TIP = 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20

CHROMATIC_NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Gesture -> (semitone offset from the selected root, chord quality).
# Ordered by finger count: 1=I, 2=IV, 3=V, 4=vi - the same four chords as
# before (C/F/G/Am in the default key), just reassigned to a simple
# "count your fingers 1-4" scheme instead of four different hand shapes.
GESTURE_TO_DEGREE = {
    "ONE_FINGER": (0, "Major"),     # I   (C)
    "TWO_FINGERS": (5, "Major"),    # IV  (F)
    "THREE_FINGERS": (7, "Major"),  # V   (G)
    "FOUR_FINGERS": (9, "Minor"),   # vi  (Am)
}


def _distance(a, b):
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def _palm_width(landmarks):
    """Index-MCP to pinky-MCP distance - roughly constant regardless of how
    open/closed the hand is, so it makes a good reference scale."""
    return _distance(landmarks[INDEX_MCP], landmarks[PINKY_MCP])


def split_hands_by_role(result):
    """
    Given a HandLandmarkerResult (up to 2 hands), return
    (chord_hand_landmarks, dynamics_hand_landmarks) - either can be None
    if that role's hand isn't currently visible.
    """
    chord_hand = None
    dynamics_hand = None

    for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        label = handedness[0].category_name  # "Left" or "Right"
        if label == config.CHORD_HAND:
            chord_hand = hand_landmarks
        elif label == config.DYNAMICS_HAND:
            dynamics_hand = hand_landmarks

    return chord_hand, dynamics_hand


def _is_thumb_extended(landmarks):
    """
    The thumb folds sideways across the palm rather than up/down, so instead
    of comparing y-coordinates like the other fingers, measure how far the
    thumb tip is from the pinky's base, in units of "palm widths"
    (config.THUMB_EXTENDED_RATIO). This stays consistent regardless of hand
    size or distance from the camera, and needs no left/right branching.
    """
    palm_width = _palm_width(landmarks)
    if palm_width < 1e-6:
        return False
    tip_distance = _distance(landmarks[THUMB_TIP], landmarks[PINKY_MCP])
    return (tip_distance / palm_width) >= config.THUMB_EXTENDED_RATIO


def _is_finger_extended(landmarks, tip_idx, pip_idx):
    """Non-thumb fingers extend upward - tip is 'above' (smaller y) the PIP joint when extended."""
    return landmarks[tip_idx].y < landmarks[pip_idx].y


def get_finger_states(hand_landmarks):
    """Return {finger_name: bool} - True if extended, for all 5 fingers of one hand."""
    return {
        "thumb": _is_thumb_extended(hand_landmarks),
        "index": _is_finger_extended(hand_landmarks, INDEX_TIP, INDEX_PIP),
        "middle": _is_finger_extended(hand_landmarks, MIDDLE_TIP, MIDDLE_PIP),
        "ring": _is_finger_extended(hand_landmarks, RING_TIP, RING_PIP),
        "pinky": _is_finger_extended(hand_landmarks, PINKY_TIP, PINKY_PIP),
    }


def classify_gesture(finger_states):
    """
    Classify by COUNTING extended fingers (thumb excluded from the count,
    and its position doesn't matter here except for FIST):
        0 non-thumb fingers, thumb also tucked -> FIST
        1 (must be the index)                  -> ONE_FINGER
        2 (must be index + middle)              -> TWO_FINGERS
        3 (must be index + middle + ring)       -> THREE_FINGERS
        4 (all four, in any combination)        -> FOUR_FINGERS
    Anything else (e.g. only the ring finger up alone) -> UNKNOWN.
    Requiring the natural counting order for 1-3 avoids odd shapes
    (like "just the pinky") being read as a valid count.
    """
    count = sum([
        finger_states["index"], finger_states["middle"],
        finger_states["ring"], finger_states["pinky"],
    ])

    if count == 0 and not finger_states["thumb"]:
        return "FIST"
    if count == 1 and finger_states["index"]:
        return "ONE_FINGER"
    if count == 2 and finger_states["index"] and finger_states["middle"]:
        return "TWO_FINGERS"
    if count == 3 and finger_states["index"] and finger_states["middle"] and finger_states["ring"]:
        return "THREE_FINGERS"
    if count == 4:
        return "FOUR_FINGERS"
    return "UNKNOWN"


def transpose_note(root_note: str, semitones: int) -> str:
    """Shift a note name up by `semitones` steps within the chromatic scale."""
    root_idx = CHROMATIC_NOTES.index(root_note)
    return CHROMATIC_NOTES[(root_idx + semitones) % 12]


def gesture_to_chord(gesture_name, root_note=None):
    """
    Map a gesture name to a chord name, transposed to `root_note` (defaults
    to config.DEFAULT_ROOT_NOTE). 'FIST' always maps to 'STOP'. Returns
    None for 'NONE'/'UNKNOWN' - the caller decides how to display that.
    """
    if gesture_name == "FIST":
        return "STOP"
    if gesture_name not in GESTURE_TO_DEGREE:
        return None

    root_note = root_note or config.DEFAULT_ROOT_NOTE
    semitones, quality = GESTURE_TO_DEGREE[gesture_name]
    chord_root = transpose_note(root_note, semitones)
    return f"{chord_root}m" if quality == "Minor" else chord_root


def calculate_dynamics(hand_landmarks):
    """
    Measure the dynamics hand's VERTICAL position in frame - raise your
    hand for louder, lower it for quieter - which is the convention most
    existing hand-tracking gesture/synth apps use (closer to a theremin
    than to "squeeze harder for louder"). Uses the wrist landmark's y
    (0.0 = top of frame, 1.0 = bottom), inverted and remapped from
    [DYNAMICS_HIGH_Y, DYNAMICS_LOW_Y] (config.py) onto [1.0, 0.0].
    """
    wrist_y = hand_landmarks[WRIST].y
    span = config.DYNAMICS_LOW_Y - config.DYNAMICS_HIGH_Y
    value = (config.DYNAMICS_LOW_Y - wrist_y) / span
    return max(0.0, min(1.0, value))


class GestureStabilizer:
    """
    Raw per-frame gesture classification can flicker for a frame or two
    (a finger sitting right on a threshold). This waits for the same raw
    gesture to repeat for config.GESTURE_STABLE_FRAMES consecutive frames
    before updating the "confirmed" gesture - a simple form of temporal
    smoothing/debouncing applied to the recognized gesture itself.
    """

    def __init__(self, required_frames: int = None):
        self._required_frames = required_frames or config.GESTURE_STABLE_FRAMES
        self._last_raw = None
        self._streak = 0
        self._confirmed = "NONE"

    def update(self, raw_gesture: str) -> str:
        if raw_gesture == self._last_raw:
            self._streak += 1
        else:
            self._last_raw = raw_gesture
            self._streak = 1

        if self._streak >= self._required_frames:
            self._confirmed = raw_gesture

        return self._confirmed
