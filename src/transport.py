"""
transport.py
-------------
Master clock for the whole instrument. Everything that needs to stay in
time (metronome clicks, loop recording start/stop, loop playback restart)
asks THIS object "what beat are we on", rather than tracking time itself.

WHY THIS IS NEEDED:
A looper without a shared clock will drift — each recorded layer would
start/stop at a slightly different moment, and layers slip out of sync
within seconds. Transport is the single source of truth for "where are we
in the bar" so the metronome and looper always agree.

BEATS_PER_LOOP = 8 (fixed, per your spec) -- one full loop is always
8 beats long, regardless of mode.
"""

import time
import numpy as np
from audio_engine import AudioEngine

BEATS_PER_LOOP = 8


class Transport:
    def __init__(self, audio_engine: AudioEngine, bpm=90):
        self.audio_engine = audio_engine
        self.bpm = bpm
        self.beat_duration = 60.0 / bpm          # seconds per beat
        self.loop_duration = self.beat_duration * BEATS_PER_LOOP

        self.running = False
        self.metronome_on = False
        self._start_time = None

        self._click_hi = self._make_click(1200.0)   # beat 1 of every bar = accented
        self._click_lo = self._make_click(800.0)     # other beats
        self._last_clicked_beat_index = -1

    @staticmethod
    def _make_click(freq, duration=0.05, sample_rate=44100):
        t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
        tone = np.sin(2 * np.pi * freq * t)
        env = np.exp(-40 * t)
        wave = (tone * env * 0.5).astype(np.float32)
        return wave

    def start(self):
        """Call once, e.g. when the program starts. Clock runs continuously
        from here -- toggling the metronome only toggles whether it CLICKS,
        not whether time is being tracked (loop sync needs the clock running
        even if you don't want to hear clicks)."""
        self.running = True
        self._start_time = time.time()

    def toggle_metronome(self):
        self.metronome_on = not self.metronome_on
        return self.metronome_on

    def elapsed(self):
        if not self.running:
            return 0.0
        return time.time() - self._start_time

    def current_beat_float(self):
        """Continuous beat position, e.g. 3.42 = a bit past beat 3."""
        return self.elapsed() / self.beat_duration

    def current_beat_index(self):
        """Integer beat number within the loop, 0-7 for an 8-beat loop."""
        return int(self.current_beat_float()) % BEATS_PER_LOOP

    def loop_position_seconds(self):
        """How far we are into the CURRENT loop cycle, in seconds.
        This is what the looper uses to know exactly where in an 8-beat
        loop we currently are, for sample-accurate record/playback sync."""
        return self.elapsed() % self.loop_duration

    def is_beat_one(self):
        """True only on the exact frame where a new loop cycle begins."""
        return self.current_beat_index() == 0

    def update(self):
        """
        Call this once per frame. Plays a metronome click exactly when we
        cross into a new beat (not continuously) -- uses a simple
        'did the integer beat number change since last frame' check.
        Returns the beat index that just started, or None if no new beat
        started this frame.
        """
        if not self.running:
            return None

        beat_idx = self.current_beat_index()
        # Use the un-modulo'd beat count to detect genuinely new beats,
        # not just "current beat index changed" (which could be ambiguous
        # right at loop boundaries).
        absolute_beat = int(self.current_beat_float())

        if absolute_beat != self._last_clicked_beat_index:
            self._last_clicked_beat_index = absolute_beat
            if self.metronome_on:
                click = self._click_hi if beat_idx == 0 else self._click_lo
                self.audio_engine.play_sound(click)
            return beat_idx

        return None