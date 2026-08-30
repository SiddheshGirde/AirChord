# AirChord 🎸✋

**AirChord** is a real-time, computer-vision-based virtual chord player. It watches
your webcam, recognizes hand gestures, and plays the matching musical chord —
no MIDI controller or instrument needed.

> **Status:** All six planned phases complete — webcam capture, hand
> detection, gesture recognition, audio playback, integration, and a
> polish/testing pass (see [Roadmap](#roadmap)). Still worth a final
> real-camera test pass on your end, especially the dynamics feel.

## Pipeline

Webcam → Frame Acquisition → OpenCV Processing → MediaPipe Hand Detection →
Landmark Extraction → Gesture Recognition (+ Scale Selection) →
Gesture-to-Chord Mapping → Chord Audio Playback → Real-time UI

## Controls

AirChord tracks **both hands** at once, each with a different job
(configurable in `config.py` via `CHORD_HAND` / `DYNAMICS_HAND`):

**Left hand — chords.** Hold up a number of fingers to pick a chord. A
chord starts **looping continuously** the moment a gesture becomes stable,
and keeps sounding until you either hold up a different number of fingers
(which cleanly replaces it) or make a fist to stop it — a brief tracking
blip on either hand won't interrupt or restart it:

| Gesture | Degree | Chord in key of C (default) |
|----------------|--------|------------------------------|
| ☝️ 1 finger    | I      | C                            |
| ✌️ 2 fingers   | IV     | F                            |
| 🤟 3 fingers   | V      | G                            |
| 🖐️ 4 fingers   | vi     | Am                           |
| ✊ Fist (0)     | —      | Stop (silences audio)        |

Counting starts at the index finger (1 = index, 2 = +middle, 3 = +ring,
4 = all four) — the thumb doesn't count towards the number, so it can rest
anywhere comfortable.

**Right hand — dynamics.** Your hand's **vertical position** controls
playback volume in real time — raise it for louder, lower it for quieter
(the same convention used by most existing hand-tracking gesture/synth
apps) — shown as a live meter bar on the right edge of the video.

**Scale / Key dropdown.** A small always-on-top window (Tkinter) lets you
pick any of the 12 chromatic root notes; the progression above transposes
to that key (e.g. "D" turns C/F/G/Am into D/G/A/Bm) — and so does the audio.

## About the audio

Chords are **synthesized in code** (`chord_player.py`) from their component
notes as additive sine waves and **looped continuously** while a gesture is
held, rather than loaded from pre-recorded files or played as a fixed-length
one-shot. This was a deliberate change from the original "one `.wav` per
chord" plan: once the scale dropdown could transpose to any of 12 keys, a
fixed 4-file library would only ever cover one of them. Synthesizing means
every key just works, with no extra audio assets and no new dependency
beyond `numpy` + `pygame` (already in `requirements.txt`).

If you'd rather use real recorded/sampled chords instead, only
`chord_player._get_sound()` needs to change — looping, dynamics-driven
volume, and stopping all stay exactly the same.

## Tech Stack

- Python 3.11
- OpenCV — video capture & image processing
- MediaPipe — hand landmark detection, via the current **MediaPipe Tasks API**
  (`HandLandmarker`). The older `mediapipe.solutions.hands` API was retired
  in 2023, so this project uses its replacement.
- NumPy — geometric gesture calculations + chord waveform synthesis
- Tkinter (Python standard library) — the scale/key dropdown panel
- Pygame — audio playback

## Project Structure

```
AirChord/
├── main.py              # Entry point - webcam + gesture + audio loop
├── config.py             # Configuration constants
├── hand_detector.py      # MediaPipe hand detection
├── gesture.py             # Finger-count, dynamics, and scale logic
├── scale_selector.py      # Tkinter key/scale dropdown
├── chord_player.py        # Synthesizes & plays chords through Pygame
├── conftest.py             # Lets tests/ import the modules above
├── tests/
│   ├── test_gesture.py       # Finger-counting, dynamics, transposition, ...
│   └── test_chord_player.py  # Note frequencies, chord voicing, waveform safety
├── requirements.txt
├── requirements-dev.txt    # Just pytest - only needed to run tests/
├── README.md
├── .gitignore
├── assets/
└── models/                  # hand_landmarker.task - auto-downloaded, not in git
```

*No `chords/` folder - since audio is synthesized rather than loaded from
files, there are no `.wav` assets to store. See "About the audio" above.*

## Testing

`tests/` covers the pure-logic parts of the project - finger-count
classification, chord/key transposition, dynamics, temporal smoothing, and
chord waveform synthesis - using synthetic hand-landmark data, so it runs
in well under a second with no camera, MediaPipe model, or audio device
needed:

```
pip install -r requirements-dev.txt
pytest
```

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
   (Tkinter ships with the standard python.org Windows installer, so no
   separate install is needed there.)

   **If VS Code underlines local imports (e.g. `chord_player`) as unresolved:**
   that's the editor's linter, not a real error - it usually means VS Code
   was opened on individual files rather than the project folder. Use
   **File → Open Folder → AirChord** so it can see sibling modules. The app
   runs correctly with `python main.py` either way.

## Running

```
python main.py
```

On the very first run, AirChord automatically downloads the MediaPipe hand
landmark model (a few MB) into `models/` — this needs an internet connection
once. After that it runs fully offline.

Two windows open: the webcam feed, and a small "AirChord - Scale" panel with
a key dropdown. Show your **left hand** with a finger-count gesture to play
a chord and move your **right hand** up/down to control volume — both
tracked at the same time, with sound through your speakers. The overlay
shows the recognized gesture and chord, the selected key, a dynamics meter,
Status (READY/PLAYING), and FPS. Press **q** (with the video window
focused) to quit.

## Concepts for your viva

Quick map from the brief's required concepts to where each one actually
lives in the code:

| Concept | Where |
|---|---|
| Video frame acquisition | `cap.read()` loop in `main.py` |
| Frame processing | `cv2.flip()` (mirroring) in `main.py` |
| RGB/BGR conversion | `hand_detector.HandDetector.detect()` converts BGR→RGB before MediaPipe (OpenCV is BGR, MediaPipe expects RGB) |
| Image resizing | `cap.set(CAP_PROP_FRAME_WIDTH/HEIGHT)` in `open_webcam()` |
| Hand/ROI detection | MediaPipe `HandLandmarker` in `hand_detector.py` |
| Landmark extraction | `result.hand_landmarks` - 21 (x, y, z) points per hand |
| Feature extraction | `gesture.get_finger_states()` (extended/folded per finger), `gesture.calculate_dynamics()` (vertical position) |
| Gesture classification | `gesture.classify_gesture()` - pure geometry/finger-counting, no ML model |
| Real-time processing | the `while True` loop in `main.py`, one pass per camera frame |
| Temporal smoothing / debouncing | `gesture.GestureStabilizer` (requires N consecutive frames before "confirming" a gesture) + the play-once-per-change trigger logic in `main.py` |
| FPS measurement | timestamp delta between frames in `main.py`'s loop |

## Roadmap

- [x] Phase 1 — Project setup + webcam capture
- [x] Phase 2 — MediaPipe hand detection & landmark visualization
- [x] Phase 3 — Two-hand gesture recognition, dynamics & scale selection
- [x] Phase 4 — Chord audio playback
- [x] Phase 5 — Full pipeline integration (built incrementally into one app throughout, rather than as a separate step)
- [x] Phase 6 — UI polish, stability, error handling, docs, and a `tests/` suite

## License

Built as a college mini-project.
