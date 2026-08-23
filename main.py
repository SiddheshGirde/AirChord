"""
main.py
--------
AirChord - Phase 2: Hand Detection & Landmark Visualization

Builds on Phase 1 by adding the next stage of the pipeline:

    ... -> Hand Detection (MediaPipe) -> Landmark Extraction -> Real-time UI

What's new in this phase:
    - Detects a hand in each frame using MediaPipe's HandLandmarker task
    - Draws all 21 hand landmarks, the skeleton connecting them, and a
      Left/Right label
    - Shows whether a hand is currently detected
    - Keeps running smoothly if no hand is in frame, or if a single
      detection call fails

Gesture recognition (finger-state logic) and audio are added in Phases 3-4.
"""

import sys
import time

import cv2

import config
import hand_detector


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


def draw_overlay(frame, fps: float, hand_detected: bool):
    """Draw the Phase 2 UI text (title, FPS, hand status, instructions)."""
    cv2.putText(frame, "AirChord - Phase 2 (Hand Detection)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    status_text = "Hand: DETECTED" if hand_detected else "Hand: NOT FOUND"
    status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
    cv2.putText(frame, status_text, (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    cv2.putText(frame, f"Press '{config.QUIT_KEY}' to quit", (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return frame


def main():
    print("Starting AirChord (Phase 2: hand detection)...")
    print(f"Press '{config.QUIT_KEY}' in the video window to quit.\n")

    try:
        cap = open_webcam(config.CAMERA_INDEX)
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        return

    try:
        detector = hand_detector.HandDetector()
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        cap.release()
        return

    prev_time = 0.0
    consecutive_failures = 0

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

            # --- Hand detection (guarded so one bad frame can't crash the app) ---
            hand_detected = False
            try:
                result = detector.detect(frame)
                hand_detected = bool(result.hand_landmarks)
                frame = detector.draw_landmarks(frame, result)
            except Exception as e:
                print(f"[WARNING] Hand detection failed on this frame: {e}")

            current_time = time.time()
            fps = 1.0 / (current_time - prev_time) if prev_time else 0.0
            prev_time = current_time

            frame = draw_overlay(frame, fps, hand_detected)
            cv2.imshow(config.WINDOW_NAME, frame)

            key_pressed = cv2.waitKey(1) & 0xFF
            window_closed = cv2.getWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
            if key_pressed == ord(config.QUIT_KEY) or window_closed:
                break
    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C).")
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam released. AirChord closed cleanly.")


if __name__ == "__main__":
    main()
