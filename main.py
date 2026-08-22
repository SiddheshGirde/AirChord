"""
main.py
--------
AirChord - Phase 1: Webcam Capture

This phase covers only the first stage of the AirChord pipeline:

    Webcam -> Video Frame Acquisition -> Frame Processing -> Real-time UI

What it does:
    - Opens the default webcam using OpenCV
    - Continuously reads frames in a loop (real-time video)
    - Mirrors the frame horizontally (natural "selfie" view)
    - Measures and displays FPS
    - Handles a missing/disconnected webcam without crashing
    - Shuts down cleanly on 'q', window close, or Ctrl+C

Hand detection (MediaPipe), gesture recognition, and audio playback are
NOT part of this phase - they are added in Phases 2-4.
"""

import sys
import time

import cv2

import config


def open_webcam(camera_index: int) -> cv2.VideoCapture:
    """Open the webcam and return a ready-to-use VideoCapture object.

    Raises RuntimeError (instead of letting OpenCV fail silently) if the
    camera can't be opened, so the caller can show a clear message and
    exit gracefully instead of crashing.
    """
    # On Windows, the default backend (MSMF) can be slow to open and prints
    # noisy warnings. DirectShow (CAP_DSHOW) tends to open faster/cleaner.
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


def draw_overlay(frame, fps: float):
    """Draw the Phase 1 UI text (title, FPS, instructions) onto the frame."""
    cv2.putText(frame, "AirChord - Phase 1 (Webcam Test)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Press '{config.QUIT_KEY}' to quit", (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return frame


def main():
    print("Starting AirChord (Phase 1: webcam test)...")
    print(f"Press '{config.QUIT_KEY}' in the video window to quit.\n")

    try:
        cap = open_webcam(config.CAMERA_INDEX)
    except RuntimeError as e:
        print(f"[ERROR] {e}")
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

            # Mirror the feed so it behaves like a mirror - natural for gestures
            frame = cv2.flip(frame, 1)

            # --- FPS calculation (time between consecutive frames) ---
            current_time = time.time()
            fps = 1.0 / (current_time - prev_time) if prev_time else 0.0
            prev_time = current_time

            frame = draw_overlay(frame, fps)
            cv2.imshow(config.WINDOW_NAME, frame)

            # Exit on 'q' key or if the window is closed via the OS close button
            key_pressed = cv2.waitKey(1) & 0xFF
            window_closed = cv2.getWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
            if key_pressed == ord(config.QUIT_KEY) or window_closed:
                break
    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C).")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam released. AirChord closed cleanly.")


if __name__ == "__main__":
    main()
