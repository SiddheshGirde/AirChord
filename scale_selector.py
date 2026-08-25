"""
scale_selector.py
-------------------
A small Tkinter side panel with a dropdown to pick the root note (key) that
the gesture-to-chord mapping transposes to (see gesture.transpose_note).

Tkinter wants to own the main thread with its own event loop, and so does
our OpenCV video loop. Rather than run Tkinter on a separate thread (which
would mean locking shared state between two threads - overkill here), this
window is "pumped" cooperatively: call `update()` once per iteration of the
main `while True` loop instead of calling `mainloop()`. Both windows then
stay responsive on a single thread.
"""

import tkinter as tk
from tkinter import ttk

import config
from gesture import CHROMATIC_NOTES


class ScaleSelector:
    """A tiny always-on-top settings window with a root-note dropdown."""

    def __init__(self):
        self._closed = False
        self._cached_note = config.DEFAULT_ROOT_NOTE

        self.root = tk.Tk()
        self.root.title("AirChord - Scale")
        self.root.geometry("260x110")
        self.root.attributes("-topmost", True)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        tk.Label(self.root, text="Scale / Key:", font=("Segoe UI", 10)).pack(pady=(14, 4))

        self._selected_note = tk.StringVar(value=config.DEFAULT_ROOT_NOTE)
        dropdown = ttk.Combobox(
            self.root,
            textvariable=self._selected_note,
            values=CHROMATIC_NOTES,
            state="readonly",       # only the listed values are selectable, no free typing
            font=("Segoe UI", 11),
            justify="center",
        )
        dropdown.pack(pady=4)

    def update(self):
        """Process pending Tkinter events. Call this once per main-loop iteration."""
        if self._closed:
            return
        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            self._closed = True

    def get_root_note(self) -> str:
        """Return the currently selected root note (e.g. 'C', 'F#')."""
        if not self._closed:
            try:
                self._cached_note = self._selected_note.get()
            except tk.TclError:
                self._closed = True
        return self._cached_note

    @property
    def is_open(self) -> bool:
        return not self._closed

    def close(self):
        """Explicitly close the settings window (safe to call more than once)."""
        if not self._closed:
            try:
                self.root.destroy()
            except tk.TclError:
                pass
            self._closed = True

    def _on_close(self):
        try:
            self._cached_note = self._selected_note.get()
        except tk.TclError:
            pass
        self._closed = True
        self.root.destroy()
