"""
main.py
--------
AirChord - Phase 4: Audio Playback

Builds on Phase 3 by adding:

    ... -> Gesture-to-Chord Mapping -> Chord Audio Playback -> Real-time UI

What's new in this phase:
    - A newly-CONFIRMED chord gesture starts that chord LOOPING continuously
      - it sustains until you show a different chord gesture (which
      replaces it) or make a fist (which stops it), not a fixed duration
    - The dynamics hand's openness continuously drives playback volume,
      live, every frame - not just at trigger time
    - A brief tracking gap on EITHER hand (gesture -> NONE/UNKNOWN, or the
      dynamics hand momentarily lost) does not interrupt the sustained
      chord or drop its volume, and does not cause an unwanted restart
      when tracking picks back up
    - Chords are SYNTHESIZED (see chord_player.py) rather than loaded from
      files, so every one of the 12 keys works with zero audio assets

Everything from Phases 1-3 (webcam, hand detection, gestures, dynamics,
scale selection) is unchanged.
"""

import sys
import time

import cv2

import config
import gesture
from hand_detector import HandDetector
from scale_selector import ScaleSelector
from chord_player import ChordPlayer


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
            f"Could not open webcam at index {camera_index}. "
            "Make sure a camera is connected and not already in use "
            "by another application."
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


def draw_overlay(frame, fps: float, state: dict, root_note: str, status: str):
    """Draw the Phase 4 UI text: title, FPS, key, hand statuses, playback status."""
    cv2.putText(frame, "AirChord - Phase 4 (Audio)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Key: {root_note}", (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    chord_color = (0, 255, 0) if state["chord_hand_visible"] else (0, 0, 255)
    if state["chord_hand_visible"]:
        chord_text = f"Chords ({config.CHORD_HAND}): {state['gesture']} -> {state['chord']}"
    else:
        chord_text = f"Chords ({config.CHORD_HAND}): no hand"
    cv2.putText(frame, chord_text, (10, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, chord_color, 2)

    dyn_color = (0, 255, 0) if state["dynamics_hand_visible"] else (0, 0, 255)
    if state["dynamics_hand_visible"]:
        dyn_text = f"Dynamics ({config.DYNAMICS_HAND}): {int(state['dynamics_value'] * 100)}%"
    else:
        dyn_text = f"Dynamics ({config.DYNAMICS_HAND}): no hand"
    cv2.putText(frame, dyn_text, (10, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, dyn_color, 2)

    status_color = (0, 255, 255) if status == "PLAYING" else (200, 200, 200)
    cv2.putText(frame, f"Status: {status}", (10, 165),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

    cv2.putText(frame, f"Press '{config.QUIT_KEY}' to quit", (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
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
    print("Starting AirChord (Phase 4: audio playback)...")
    print(f"Chord hand: {config.CHORD_HAND}   Dynamics hand: {config.DYNAMICS_HAND}")
    print(f"Press '{config.QUIT_KEY}' in the video window to quit.\n")

    try:
        cap = open_webcam(config.CAMERA_INDEX)
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        return

    try:
        detector = HandDetector()
    except RuntimeError as e:
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
