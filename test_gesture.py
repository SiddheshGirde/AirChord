"""
tests/test_gesture.py
-----------------------
Unit tests for gesture.py's pure-geometry logic: finger counting,
gesture classification, dynamics, key transposition, temporal smoothing,
and hand-role splitting. No camera, MediaPipe, or audio needed - these
all run on plain Python objects.

Run with:
    pip install pytest
    pytest
"""

import pytest

import gesture


class FakeLandmark:
    """Stand-in for a MediaPipe NormalizedLandmark: just needs .x/.y/.z."""

    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z


def make_hand(thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext):
    """Build a synthetic 21-point hand with the given fingers extended/folded."""
    wrist = FakeLandmark(0.5, 0.9)
    thumb_cmc = FakeLandmark(0.47, 0.75)
    thumb_mcp = FakeLandmark(0.44, 0.72)
    index_mcp = FakeLandmark(0.43, 0.70)
    middle_mcp = FakeLandmark(0.50, 0.68)
    ring_mcp = FakeLandmark(0.55, 0.70)
    pinky_mcp = FakeLandmark(0.57, 0.72)

    def finger_points(mcp_x, extended):
        if extended:
            return [FakeLandmark(mcp_x, 0.55), FakeLandmark(mcp_x, 0.40), FakeLandmark(mcp_x, 0.25)]
        return [FakeLandmark(mcp_x, 0.60), FakeLandmark(mcp_x, 0.65), FakeLandmark(mcp_x, 0.68)]

    thumb_tip = FakeLandmark(0.25, 0.65) if thumb_ext else FakeLandmark(0.46, 0.69)
    thumb_ip = FakeLandmark(0.35, 0.68) if thumb_ext else FakeLandmark(0.45, 0.70)

    index_pip, index_dip, index_tip = finger_points(0.43, index_ext)
    middle_pip, middle_dip, middle_tip = finger_points(0.50, middle_ext)
    ring_pip, ring_dip, ring_tip = finger_points(0.55, ring_ext)
    pinky_pip, pinky_dip, pinky_tip = finger_points(0.58, pinky_ext)

    return [
        wrist,
        thumb_cmc, thumb_mcp, thumb_ip, thumb_tip,
        index_mcp, index_pip, index_dip, index_tip,
        middle_mcp, middle_pip, middle_dip, middle_tip,
        ring_mcp, ring_pip, ring_dip, ring_tip,
        pinky_mcp, pinky_pip, pinky_dip, pinky_tip,
    ]


def classify(thumb, index, middle, ring, pinky):
    hand = make_hand(thumb, index, middle, ring, pinky)
    return gesture.classify_gesture(gesture.get_finger_states(hand))


# --- classify_gesture: the five valid finger counts ---

def test_fist_is_zero_fingers():
    assert classify(False, False, False, False, False) == "FIST"

def test_one_finger():
    assert classify(False, True, False, False, False) == "ONE_FINGER"

def test_two_fingers():
    assert classify(False, True, True, False, False) == "TWO_FINGERS"

def test_three_fingers():
    assert classify(False, True, True, True, False) == "THREE_FINGERS"

def test_four_fingers():
    assert classify(False, True, True, True, True) == "FOUR_FINGERS"

def test_four_fingers_with_thumb_out_still_counts():
    # Thumb position doesn't matter for the 1-4 count, only for FIST.
    assert classify(True, True, True, True, True) == "FOUR_FINGERS"


# --- classify_gesture: shapes that should NOT match any count ---

def test_only_ring_finger_is_unknown():
    assert classify(False, False, False, True, False) == "UNKNOWN"

def test_index_and_ring_without_middle_is_unknown():
    assert classify(False, True, False, True, False) == "UNKNOWN"

def test_thumbs_up_alone_is_unknown():
    assert classify(True, False, False, False, False) == "UNKNOWN"


# --- gesture_to_chord: finger-count -> chord mapping, and transposition ---

def test_default_key_chord_mapping():
    assert gesture.gesture_to_chord("ONE_FINGER", "C") == "C"
    assert gesture.gesture_to_chord("TWO_FINGERS", "C") == "F"
    assert gesture.gesture_to_chord("THREE_FINGERS", "C") == "G"
    assert gesture.gesture_to_chord("FOUR_FINGERS", "C") == "Am"
    assert gesture.gesture_to_chord("FIST", "C") == "STOP"

def test_transposed_key_of_d():
    assert gesture.gesture_to_chord("ONE_FINGER", "D") == "D"
    assert gesture.gesture_to_chord("TWO_FINGERS", "D") == "G"
    assert gesture.gesture_to_chord("THREE_FINGERS", "D") == "A"
    assert gesture.gesture_to_chord("FOUR_FINGERS", "D") == "Bm"

def test_unknown_and_none_have_no_chord():
    assert gesture.gesture_to_chord("UNKNOWN", "C") is None
    assert gesture.gesture_to_chord("NONE", "C") is None

def test_transpose_note_wraps_around_the_octave():
    assert gesture.transpose_note("B", 1) == "C"
    assert gesture.transpose_note("C", -1) == "B"

@pytest.mark.parametrize("root,expected_progression", [
    ("C", ["C", "F", "G", "Am"]),
    ("G", ["G", "C", "D", "Em"]),
    ("D", ["D", "G", "A", "Bm"]),
])
def test_full_progression_across_keys(root, expected_progression):
    gestures = ["ONE_FINGER", "TWO_FINGERS", "THREE_FINGERS", "FOUR_FINGERS"]
    got = [gesture.gesture_to_chord(g, root) for g in gestures]
    assert got == expected_progression


# --- calculate_dynamics: vertical-position volume ---

def _hand_with_wrist_y(y):
    return [FakeLandmark(0.5, y)] + [FakeLandmark(0.5, 0.5)] * 20

def test_dynamics_hand_near_top_is_loud():
    assert gesture.calculate_dynamics(_hand_with_wrist_y(0.15)) == pytest.approx(1.0, abs=0.01)

def test_dynamics_hand_near_bottom_is_quiet():
    assert gesture.calculate_dynamics(_hand_with_wrist_y(0.85)) == pytest.approx(0.0, abs=0.01)

def test_dynamics_clamps_beyond_the_configured_range():
    assert gesture.calculate_dynamics(_hand_with_wrist_y(0.02)) == pytest.approx(1.0, abs=0.01)
    assert gesture.calculate_dynamics(_hand_with_wrist_y(0.98)) == pytest.approx(0.0, abs=0.01)

def test_dynamics_is_monotonic_with_height():
    values = [gesture.calculate_dynamics(_hand_with_wrist_y(y)) for y in (0.2, 0.35, 0.5, 0.65, 0.8)]
    assert values == sorted(values, reverse=True)


# --- GestureStabilizer: temporal smoothing / debouncing ---

def test_stabilizer_waits_for_consecutive_frames():
    stabilizer = gesture.GestureStabilizer(required_frames=5)
    result = None
    for _ in range(4):
        result = stabilizer.update("ONE_FINGER")
    assert result != "ONE_FINGER"
    assert stabilizer.update("ONE_FINGER") == "ONE_FINGER"

def test_stabilizer_resets_streak_on_change():
    stabilizer = gesture.GestureStabilizer(required_frames=3)
    stabilizer.update("ONE_FINGER")
    stabilizer.update("ONE_FINGER")
    stabilizer.update("TWO_FINGERS")  # streak resets here
    assert stabilizer.update("TWO_FINGERS") != "TWO_FINGERS"
    assert stabilizer.update("TWO_FINGERS") == "TWO_FINGERS"


# --- split_hands_by_role ---

class _FakeCategory:
    def __init__(self, name):
        self.category_name = name

class _FakeResult:
    def __init__(self, hands):
        self.hand_landmarks = [h[0] for h in hands]
        self.handedness = [[_FakeCategory(h[1])] for h in hands]

def test_split_hands_by_role_assigns_correctly():
    left_hand = make_hand(False, True, False, False, False)
    right_hand = make_hand(True, True, True, True, True)
    result = _FakeResult([(left_hand, "Left"), (right_hand, "Right")])
    chord_hand, dynamics_hand = gesture.split_hands_by_role(result)
    assert chord_hand is left_hand
    assert dynamics_hand is right_hand

def test_split_hands_by_role_handles_one_hand_missing():
    right_hand = make_hand(True, True, True, True, True)
    result = _FakeResult([(right_hand, "Right")])
    chord_hand, dynamics_hand = gesture.split_hands_by_role(result)
    assert chord_hand is None
    assert dynamics_hand is right_hand

def test_split_hands_by_role_handles_no_hands():
    result = _FakeResult([])
    chord_hand, dynamics_hand = gesture.split_hands_by_role(result)
    assert chord_hand is None
    assert dynamics_hand is None
