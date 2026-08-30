"""
tests/test_chord_player.py
-----------------------------
Unit tests for chord_player.py's synthesis math: note frequencies, chord
voicing, and waveform generation. None of these touch an actual audio
device, but chord_player.py imports pygame at module level, so these
tests are skipped (not failed) if pygame isn't installed yet.

Run with:
    pip install pytest
    pytest
"""

import numpy as np
import pytest

pytest.importorskip("pygame", reason="pygame not installed - run: pip install -r requirements.txt")

import config
import chord_player as cp


def test_c4_is_middle_c():
    assert cp._note_frequency("C") == pytest.approx(261.63, abs=0.1)

def test_a4_is_concert_pitch():
    assert cp._note_frequency("A") == pytest.approx(440.0, abs=0.1)

def test_octave_up_doubles_frequency():
    assert cp._note_frequency("C", octave_shift=1) == pytest.approx(cp._note_frequency("C") * 2, abs=0.1)

def test_octave_down_halves_frequency():
    assert cp._note_frequency("C", octave_shift=-1) == pytest.approx(cp._note_frequency("C") / 2, abs=0.1)


@pytest.mark.parametrize("chord,expected", [
    ("C", [("C", -1), ("C", 0), ("E", 0), ("G", 0)]),
    ("Am", [("A", -1), ("A", 0), ("C", 0), ("E", 0)]),
    ("D", [("D", -1), ("D", 0), ("F#", 0), ("A", 0)]),
    ("Bm", [("B", -1), ("B", 0), ("D", 0), ("F#", 0)]),
    ("G", [("G", -1), ("G", 0), ("B", 0), ("D", 0)]),
])
def test_chord_notes_match_music_theory(chord, expected):
    assert cp._chord_notes(chord) == expected


def test_every_key_and_quality_resolves_without_error():
    """All 12 roots x Major/Minor (24 combinations) should resolve cleanly -
    this is what makes the scale dropdown work in every key."""
    for root in cp.CHROMATIC_NOTES:
        for chord_name in (root, root + "m"):
            notes = cp._chord_notes(chord_name)
            for note_name, _ in notes:
                cp._note_frequency(note_name)  # raises if the note name is bad


def test_waveform_is_sustained_not_decaying():
    """Peak amplitude should stay roughly constant across the buffer - the
    decay envelope was deliberately removed so chords can loop/sustain
    until explicitly stopped, instead of fading out on their own."""
    waveform = cp._chord_to_waveform("C")
    n = len(waveform)
    start_peak = np.max(np.abs(waveform[int(n * 0.05):int(n * 0.10)]))
    end_peak = np.max(np.abs(waveform[int(n * 0.85):int(n * 0.90)]))
    assert end_peak / start_peak > 0.85  # within ~15%, not decayed away


def test_waveform_shape_and_safety():
    waveform = cp._chord_to_waveform("C")
    expected_len = int(cp.SAMPLE_RATE * config.CHORD_DURATION_SECONDS)
    assert waveform.shape == (expected_len, 2)
    assert waveform.dtype == np.int16
    assert np.max(np.abs(waveform)) < 32767  # headroom, no clipping
    assert np.max(np.abs(waveform)) > 1000   # actually audible, not silent


def test_waveform_fades_at_loop_seam_to_avoid_a_click():
    waveform = cp._chord_to_waveform("C")
    assert abs(int(waveform[0][0])) < 500
    assert abs(int(waveform[-1][0])) < 500
