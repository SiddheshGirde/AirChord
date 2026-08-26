# AirChord 🎸✋

**AirChord** is a real-time, computer-vision-based virtual chord player. It watches
your webcam, recognizes hand gestures, and plays the matching musical chord —
no MIDI controller or instrument needed.

> **Status:** Work in progress — currently on **Phase 4: Audio Playback**.
> The full pipeline now runs end to end. UI polish and documentation are
> finished off next (see [Roadmap](#roadmap)).

## Pipeline

Webcam → Frame Acquisition → OpenCV Processing → MediaPipe Hand Detection →
Landmark Extraction → Gesture Recognition (+ Scale Selection) →
Gesture-to-Chord Mapping → Chord Audio Playback → Real-time UI

## Controls

AirChord tracks **both hands** at once, each with a different job
(configurable in `config.py` via `CHORD_HAND` / `DYNAMICS_HAND`):

**Left hand — chords.** Finger shape → scale degree, transposed to whichever
key you've selected. A chord starts **looping continuously** the moment a
gesture becomes stable, and keeps sounding until you either show a
different chord gesture (which cleanly replaces it) or make a fist to
stop it — a brief tracking blip on either hand won't interrupt or restart it:

| Gesture | Degree | Chord in key of C (default) |
|---------------------------------|--------|------------------------------|
| ☝️ One finger                   | I      | C                            |
| ✌️ Two fingers                  | V      | G                            |
| 🤟 Rock on (thumb+index+pinky)  | vi     | Am                           |
| 🖐️ Open palm                    | IV     | F                            |
| ✊ Fist                          | —      | Stop (silences audio)        |

**Right hand — dynamics.** How open/spread your hand is continuously
controls playback volume in real time (closed fist ≈ quiet, fully open ≈
loud) — shown as a live meter bar on the right edge of the video.

**Scale / Key dropdown.** A small always-on-top window (Tkinter) lets you
pick any of the 12 chromatic root notes; the progression above transposes
to that key (e.g. "D" turns C/G/Am/F into D/A/Bm/G) — and so does the audio.

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
`chord_player._get_sound()` needs to change — playback, debouncing,
dynamics-driven volume, and stopping all stay exactly the same.

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
├── gesture.py             # Finger-state, dynamics, and scale logic
├── scale_selector.py      # Tkinter key/scale dropdown
├── chord_player.py        # Synthesizes & plays chords through Pygame
├── requirements.txt
├── README.md
├── .gitignore
├── assets/
└── models/                  # hand_landmarker.task - auto-downloaded, not in git
```

*No `chords/` folder - since audio is synthesized rather than loaded from
files, there are no `.wav` assets to store. See "About the audio" above.*

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

## Running (Phase 4)

```
python main.py
```

On the very first run, AirChord automatically downloads the MediaPipe hand
landmark model (a few MB) into `models/` — this needs an internet connection
once. After that it runs fully offline.

Two windows open: the webcam feed, and a small "AirChord - Scale" panel with
a key dropdown. Show your **left hand** to play a chord gesture and your
**right hand** open/closed to control volume — both tracked at the same
time, with sound coming through your speakers. The overlay shows FPS, the
selected key, the recognized gesture and chord, a dynamics meter, and a
Status line (READY / PLAYING). Press **q** (with the video window focused)
to quit.

## Roadmap

- [x] Phase 1 — Project setup + webcam capture
- [x] Phase 2 — MediaPipe hand detection & landmark visualization
- [x] Phase 3 — Two-hand gesture recognition, dynamics & scale selection
- [ ] Phase 4 — Chord audio playback
- [ ] Phase 5 — Full pipeline integration
- [ ] Phase 6 — UI polish, stability, error handling, docs

## License

Built as a college mini-project.
