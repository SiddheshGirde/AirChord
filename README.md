# AirChord 🎸✋

**AirChord** is a real-time, computer-vision-based virtual chord player. It watches
your webcam, recognizes a hand gesture, and plays the matching musical chord —
no MIDI controller or instrument needed.

> **Status:** Work in progress — currently on **Phase 1: Webcam Capture**.
> Hand detection, gesture recognition, and audio playback are added in later
> phases (see [Roadmap](#roadmap)).

## Pipeline

Webcam → Frame Acquisition → OpenCV Processing → MediaPipe Hand Detection →
Landmark Extraction → Gesture Recognition → Gesture-to-Chord Mapping →
Audio Playback (Pygame) → Real-time UI

## Gestures → Chords

| Gesture | Chord   |
|---------|---------|
| ☝️      | C Major |
| ✌️      | G Major |
| 🤟      | A Minor |
| 🖐️      | F Major |
| ✊      | Stop    |

## Tech Stack

- Python 3.11
- OpenCV — video capture & image processing
- MediaPipe — hand landmark detection *(Phase 2)*
- NumPy — geometric calculations for gesture logic
- Pygame — audio playback *(Phase 4)*

## Project Structure

Full structure we're building toward (created incrementally — see [Roadmap](#roadmap)):

```
AirChord/
├── main.py              # Entry point - webcam loop (Phase 1)
├── config.py             # Configuration constants (Phase 1)
├── requirements.txt
├── README.md
├── .gitignore
├── hand_detector.py      # MediaPipe hand detection (Phase 2)
├── gesture.py             # Finger-state / gesture recognition (Phase 3)
├── chord_player.py        # Pygame audio playback (Phase 4)
├── chords/                  # Chord audio files (Phase 4)
└── assets/
```

*Only the Phase 1 files (`main.py`, `config.py`, `requirements.txt`, `README.md`,
`.gitignore`) exist right now.*

## Setup

1. Clone the repo and enter it:
   ```
   git clone <your-repo-url>
   cd AirChord
   ```
2. Create and activate a virtual environment (recommended):
   ```
   python -m venv venv
   venv\Scripts\activate       # Windows
   source venv/bin/activate    # macOS/Linux
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running (Phase 1)

```
python main.py
```

A window opens showing your mirrored webcam feed with an FPS counter.
Press **q** in the video window to quit.

## Roadmap

- [x] Phase 1 — Project setup + webcam capture
- [ ] Phase 2 — MediaPipe hand detection & landmark visualization
- [ ] Phase 3 — Finger-state & gesture recognition
- [ ] Phase 4 — Chord audio playback
- [ ] Phase 5 — Full pipeline integration
- [ ] Phase 6 — UI polish, stability, error handling, docs

## License

Built as a college mini-project.
