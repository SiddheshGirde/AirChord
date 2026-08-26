"""
chord_player.py
-----------------
Turns a chord name (e.g. 'C', 'Bm') into sound and plays it through Pygame,
LOOPED continuously until explicitly changed or stopped.

DESIGN NOTE: chords are SYNTHESIZED on the fly from their component notes
(additive sine waves) rather than loaded from pre-recorded files. This is
a deliberate change from the original "one .wav per chord" plan: Phase 3
added a 12-key scale dropdown, and a fixed 4-file library only covers one
key. Synthesizing means every key just works, with zero extra audio assets
and no new dependency beyond numpy + pygame (already in requirements.txt).

Swapping to real recorded/sampled chords later only requires changing
_get_sound() below - everything else (looping, dynamics volume, stopping)
stays exactly the same.
"""

import numpy as np
import pygame

import config
from gesture import CHROMATIC_NOTES

SAMPLE_RATE = 44100
C4_FREQUENCY = 261.63  # Hz, standard "middle C"

NOTE_TO_SEMITONE = {name: i for i, name in enumerate(CHROMATIC_NOTES)}


def _note_frequency(note_name: str, octave_shift: int = 0) -> float:
    """Frequency in Hz for a note name relative to C4, optionally shifted by octaves."""
    semitones = NOTE_TO_SEMITONE[note_name] + (octave_shift * 12)
    return C4_FREQUENCY * (2 ** (semitones / 12))


def _chord_notes(chord_name: str):
    """
    Return [(note_name, octave_shift), ...] for a chord's triad, e.g.
    'Am' -> [('A', -1), ('A', 0), ('C', 0), ('E', 0)]. A major triad is
    root/major-third/fifth (+0/+4/+7 semitones); minor is +0/+3/+7. The
    root is doubled an octave down for a fuller, more "chord-like" sound.
    """
    is_minor = chord_name.endswith("m")
    root = chord_name[:-1] if is_minor else chord_name
    root_semitone = NOTE_TO_SEMITONE[root]
    third_offset = 3 if is_minor else 4

    root_note = CHROMATIC_NOTES[root_semitone]
    third_note = CHROMATIC_NOTES[(root_semitone + third_offset) % 12]
    fifth_note = CHROMATIC_NOTES[(root_semitone + 7) % 12]

    return [(root_note, -1), (root_note, 0), (third_note, 0), (fifth_note, 0)]


def _chord_to_waveform(chord_name: str) -> np.ndarray:
    """
    Synthesize one loop-buffer's worth of a sustained chord: a sum of sine
    waves (one per note, plus a soft octave-up harmonic per note for
    warmth), at a STEADY volume - no decay - since this buffer is looped
    continuously by play_chord(). A very short fade in/out (a few ms) is
    still applied at the very start/end purely to avoid an audible "click"
    at the loop seam; it does not make the sound fade out overall.
    Returns int16 stereo PCM samples.
    """
    duration = config.CHORD_DURATION_SECONDS
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    waveform = np.zeros_like(t)

    notes = _chord_notes(chord_name)
    for note_name, octave_shift in notes:
        freq = _note_frequency(note_name, octave_shift)
        waveform += np.sin(2 * np.pi * freq * t)
        waveform += 0.25 * np.sin(2 * np.pi * (freq * 2) * t)  # octave-up harmonic
    waveform /= len(notes)

    # Tiny fade at the very start/end only, to prevent a click at the loop
    # seam - this is NOT an overall decay, the sustained volume in between
    # stays constant so the loop sounds like a held chord, not a fading one.
    fade_samples = int(0.005 * SAMPLE_RATE)  # 5 ms
    envelope = np.ones_like(waveform)
    envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
    envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
    waveform *= envelope

    stereo = np.column_stack((waveform, waveform))
    return (stereo * 32767 * 0.4).astype(np.int16)  # 0.4 headroom - avoids clipping


class ChordPlayer:
    """Synthesizes and plays AirChord's chords through Pygame, with volume control."""

    def __init__(self):
        pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2)
        self._sound_cache = {}   # chord_name -> pygame.mixer.Sound, built once and reused
        self._current_channel = None

    def _get_sound(self, chord_name: str) -> pygame.mixer.Sound:
        if chord_name not in self._sound_cache:
            waveform = _chord_to_waveform(chord_name)
            self._sound_cache[chord_name] = pygame.sndarray.make_sound(waveform)
        return self._sound_cache[chord_name]

    def play_chord(self, chord_name: str, volume: float = 1.0):
        """Start looping `chord_name` (e.g. 'D', 'Bm') continuously at 0.0-1.0
        volume, replacing whatever was playing before. It keeps sounding until
        play_chord() is called again with a different chord, or stop() is
        called. Playback errors are caught and logged, never raised."""
        try:
            if self._current_channel is not None:
                self._current_channel.stop()
            sound = self._get_sound(chord_name)
            sound.set_volume(max(0.0, min(1.0, volume)))
            self._current_channel = sound.play(loops=-1)  # -1 = loop forever
        except Exception as e:
            print(f"[WARNING] Could not play chord '{chord_name}': {e}")

    def set_volume(self, volume: float):
        """Live-update the volume of whatever's currently playing (drives the dynamics hand)."""
        if self._current_channel is not None:
            try:
                self._current_channel.set_volume(max(0.0, min(1.0, volume)))
            except Exception:
                pass

    def is_playing(self) -> bool:
        return self._current_channel is not None and self._current_channel.get_busy()

    def stop(self):
        """Stop all AirChord audio immediately (the FIST/Stop gesture)."""
        pygame.mixer.stop()
        self._current_channel = None

    def close(self):
        """Release the Pygame mixer's resources."""
        try:
            pygame.mixer.quit()
        except Exception:
            pass
