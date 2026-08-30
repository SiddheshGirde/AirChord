"""
main.py
--------
AirChord - real-time webcam gesture-to-chord player.

Full pipeline:
    Webcam -> Frame Acquisition -> OpenCV Processing -> MediaPipe Hand
    Detection -> Landmark Extraction -> Gesture Recognition (+ Scale
    Selection) -> Gesture-to-Chord Mapping -> Chord Audio Playback ->
    Real-time UI

How it plays:
    - LEFT hand shows a finger-count gesture (0-4 fingers) that selects a
      chord; RIGHT hand's vertical position controls volume (config.py's
      CHORD_HAND / DYNAMICS_HAND can swap which physical hand does which)
    - A newly-CONFIRMED gesture starts that chord LOOPING continuously -
      it sustains until you show a different gesture (replaces it
      cleanly) or make a fist (stops it) - not a fixed duration
    - A brief tracking gap on either hand does not interrupt the
      sustained chord, drop its volume, or cause an unwanted restart
    - Chords are SYNTHESIZED (see chord_player.py) rather than loaded
      from files, so all 12 keys work with zero audio assets
"""

import sys
import time

import cv2

import config
import gesture
from hand_detector import HandDetector
from scale_selector import ScaleSelector
from chord_player import ChordPlayer


def _validate_config():
    """Catch a common misconfiguration early with a clear message, instead
    of both hands silently fighting over the same role at runtime."""
    if config.CHORD_HAND == config.DYNAMICS_HAND:
        raise RuntimeError(
            f"config.CHORD_HAND and config.DYNAMICS_HAND are both set to "
            f"'{config.CHORD_HAND}' - they need to be different hands "
            f"('Left' and 'Right', in either order)."
        )


def open_webcam(camera_index: int) -> cv2.VideoCapture:
    """Open the webcam and return a ready-to-use VideoCapture object.

    Raises RuntimeError (instead of letting OpenCV fail silently) if the
    camera can't be opened, so the caller can show a clear message and
    exit gracefully instead of crashing.
    """
    if sys.platform == "win32":
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam at index {camera_index}. Check that: "
            "a camera is connected; it isn't already in use by another "
            "app (Zoom, Teams, another AirChord instance, ...); and on "
            "Windows, Settings > Privacy & security > Camera has access "
            "turned on for desktop apps."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    return cap


def _default_state():
    return {
        "chord_hand_visible": False,
        "dynamics_hand_visible": False,
        "gesture": "NONE",
        "chord": "-",
        "dynamics_value": 0.0,
    }


def process_frame(frame, detector, stabilizer, root_note, last_dynamics_value=0.0):
    """
    Run hand detection + gesture/dynamics interpretation on one frame.
    Returns (frame_with_landmarks_drawn, state_dict).
    """
    state = _default_state()
    # Preserve dynamics volume across a brief dynamics-hand tracking gap
    # instead of snapping to 0 - a one-frame blip shouldn't drop the volume.
    state["dynamics_value"] = last_dynamics_value

    result = detector.detect(frame)
    frame = detector.draw_landmarks(frame, result)

    chord_hand, dynamics_hand = gesture.split_hands_by_role(result)

    if chord_hand is not None:
        state["chord_hand_visible"] = True
        finger_states = gesture.get_finger_states(chord_hand)
        raw_gesture = gesture.classify_gesture(finger_states)
        state["gesture"] = stabilizer.update(raw_gesture)
    else:
        # Let the stabilizer know the hand is gone so it releases a stale gesture.
        state["gesture"] = stabilizer.update("NONE")

    state["chord"] = gesture.gesture_to_chord(state["gesture"], root_note) or "-"

    if dynamics_hand is not None:
        state["dynamics_hand_visible"] = True
        state["dynamics_value"] = gesture.calculate_dynamics(dynamics_hand)

    return frame, state


def _format_gesture_display(gesture_name: str) -> str:
    """'TWO_FINGERS' -> 'TWO FINGERS', 'NONE' -> 'NONE' - readable on-screen labels."""
    return gesture_name.replace("_", " ")


def _format_chord_display(chord_name: str) -> str:
    """'Am' -> 'A MINOR', 'C' -> 'C MAJOR', 'STOP'/'-' unchanged."""
    if chord_name in ("-", "STOP"):
        return chord_name
    if chord_name.endswith("m"):
        return f"{chord_name[:-1]} MINOR"
    return f"{chord_name} MAJOR"


def draw_overlay(frame, fps: float, state: dict, root_note: str, status: str):
    """Draw the UI text, styled after the project brief's AIRCHORD / dashes /
    labeled-fields mockup, plus the extra fields this project added (key,
    per-hand visibility, dynamics)."""
    w = frame.shape[1]

    def text(label, y, scale=0.6, color=(0, 255, 0), thickness=2):
        cv2.putText(frame, label, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

    text("AIRCHORD", 30, 0.8, (0, 255, 0), 2)
    cv2.line(frame, (10, 42), (w - 45, 42), (120, 120, 120), 1)

    chord_color = (0, 255, 0) if state["chord_hand_visible"] else (0, 0, 255)
    gesture_label = (_format_gesture_display(state["gesture"]) if state["chord_hand_visible"]
                      else "no hand")
    text(f"Gesture ({config.CHORD_HAND}): {gesture_label}", 68, color=chord_color)
    text(f"Chord: {_format_chord_display(state['chord'])}   Key: {root_note}", 93, color=(255, 255, 0))

    dyn_color = (0, 255, 0) if state["dynamics_hand_visible"] else (0, 0, 255)
    dyn_label = f"{int(state['dynamics_value'] * 100)}%" if state["dynamics_hand_visible"] else "no hand"
    text(f"Dynamics ({config.DYNAMICS_HAND}): {dyn_label}", 118, color=dyn_color)

    status_color = (0, 255, 255) if status == "PLAYING" else (200, 200, 200)
    text(f"Status: {status}", 143, color=status_color)
    text(f"FPS: {int(fps)}", 168, color=(0, 255, 0))

    cv2.line(frame, (10, 180), (w - 45, 180), (120, 120, 120), 1)
    text(f"Press '{config.QUIT_KEY}' to quit", frame.shape[0] - 15, scale=0.5,
         color=(200, 200, 200), thickness=1)
    return frame


def draw_dynamics_meter(frame, value: float):
    """Draw a vertical level-meter bar on the right edge of the frame."""
    h, w = frame.shape[:2]
    bar_x1, bar_x2 = w - 40, w - 15
    bar_top, bar_bottom = 50, h - 50

    cv2.rectangle(frame, (bar_x1, bar_top), (bar_x2, bar_bottom), (200, 200, 200), 2)

    fill_height = int((bar_bottom - bar_top) * value)
    fill_top = bar_bottom - fill_height
    if fill_height > 0:
        cv2.rectangle(frame, (bar_x1, fill_top), (bar_x2, bar_bottom), (0, 255, 255), -1)

    cv2.putText(frame, "VOL", (bar_x1 - 8, bar_top - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    return frame


def main():
    print("Starting AirChord...")
    print(f"Chord hand: {config.CHORD_HAND}   Dynamics hand: {config.DYNAMICS_HAND}")
    print(f"Press '{config.QUIT_KEY}' in the video window to quit.\n")

    try:
        _validate_config()
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        return

    try:
        cap = open_webcam(config.CAMERA_INDEX)
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        return

    try:
        detector = HandDetector()
    except Exception as e:
        print(f"[ERROR] {e}")
        cap.release()
        return

    try:
        scale_selector = ScaleSelector()
    except Exception as e:
        print(f"[WARNING] Could not open the scale-selector window ({e}). "
              f"Continuing with a fixed key of {config.DEFAULT_ROOT_NOTE}.")
        scale_selector = None

    try:
        player = ChordPlayer()
    except Exception as e:
        print(f"[WARNING] Could not start audio playback ({e}). "
              "Continuing without sound - gestures and chords still display normally.")
        player = None

    stabilizer = gesture.GestureStabilizer()

    prev_time = 0.0
    consecutive_failures = 0
    last_played_gesture = None
    last_dynamics_value = 0.0

    try:
        while True:
            ret, frame = cap.read()

            if not ret or frame is None:
                consecutive_failures += 1
                print("[WARNING] Failed to grab frame from webcam. Retrying...")
                if consecutive_failures >= config.MAX_CONSECUTIVE_FRAME_FAILURES:
                    print("[ERROR] Too many failed frame reads in a row. "
                          "Is the webcam disconnected?")
                    break
                continue
            consecutive_failures = 0

            frame = cv2.flip(frame, 1)

            root_note = scale_selector.get_root_note() if scale_selector else config.DEFAULT_ROOT_NOTE

            try:
                frame, state = process_frame(frame, detector, stabilizer, root_note, last_dynamics_value)
            except Exception as e:
                print(f"[WARNING] Hand detection/gesture recognition failed on this frame: {e}")
                state = _default_state()
                state["dynamics_value"] = last_dynamics_value
            last_dynamics_value = state["dynamics_value"]

            # Chords now SUSTAIN continuously until you either show a different
            # chord gesture or make a fist to stop - not a one-shot strum. So:
            #   - a real chord gesture different from what's already looping
            #     -> start looping the new one (replaces the old one cleanly)
            #   - FIST (and we're not already stopped) -> stop looping
            #   - NONE/UNKNOWN (e.g. a brief tracking gap) -> do NOT touch
            #     last_played_gesture at all, so whatever's looping keeps
            #     looping uninterrupted, and doesn't restart when the same
            #     gesture reappears a frame later.
            if player:
                if state["gesture"] in gesture.GESTURE_TO_DEGREE and state["gesture"] != last_played_gesture:
                    last_played_gesture = state["gesture"]
                    player.play_chord(state["chord"], volume=state["dynamics_value"])
                elif state["gesture"] == "FIST" and last_played_gesture != "FIST":
                    last_played_gesture = "FIST"
                    player.stop()

                # Dynamics hand drives volume live, every frame, of whatever's looping.
                player.set_volume(state["dynamics_value"])

            current_time = time.time()
            fps = 1.0 / (current_time - prev_time) if prev_time else 0.0
            prev_time = current_time

            status = "PLAYING" if (player and player.is_playing()) else "READY"
            frame = draw_overlay(frame, fps, state, root_note, status)
            frame = draw_dynamics_meter(frame, state["dynamics_value"])
            cv2.imshow(config.WINDOW_NAME, frame)

            if scale_selector:
                scale_selector.update()

            key_pressed = cv2.waitKey(1) & 0xFF
            window_closed = cv2.getWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
            if key_pressed == ord(config.QUIT_KEY) or window_closed:
                break
    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C).")
    finally:
        detector.close()
        if scale_selector:
            scale_selector.close()
        if player:
            player.close()
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam released. AirChord closed cleanly.")


if __name__ == "__main__":
    main()
