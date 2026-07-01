"""
looper.py
----------
One looper "slot" per mode (guitar / drums / strings), each holding its
own fixed-length (8-beat) recording. Recording is QUANTIZED to the
transport: pressing record doesn't start capturing immediately -- it ARMS,
and actual recording starts on the next beat-1. This is what makes loops
line up musically instead of starting wherever your finger happened to
press the key.

HOW PLAYBACK WORKS:
A loop slot doesn't store audio -- it stores a list of (beat_position,
sound_array) events. On every transport loop cycle, we replay those events
at the same beat positions by feeding them to the AudioEngine again. This
is simpler and far more flexible than recording/looping a raw audio buffer
(it also means a recorded chord and a recorded drum hit both just become
"events", regardless of which mode produced them).
"""

from transport import Transport, BEATS_PER_LOOP


class LoopSlot:
    """One mode's loop (e.g. the Guitar loop, or the Drums loop)."""

    def __init__(self, name):
        self.name = name
        self.events = []        # list of (beat_position_float, wave_array)
        self.has_content = False
        self.is_recording = False
        self._armed = False
        self._last_played_cycle = -1

    def arm_record(self):
        """Call when the user presses 'record' for this mode. Recording
        doesn't start yet -- it waits for the next beat-1."""
        self._armed = True

    def stop_record(self):
        self.is_recording = False
        self._armed = False

    def clear(self):
        self.events = []
        self.has_content = False
        self.is_recording = False
        self._armed = False

    def capture_event(self, beat_position, wave_array):
        """Call this every time a sound is triggered in this mode (e.g. a
        strum, a drum hit). If we're currently recording, store it so it
        replays on future loop cycles."""
        if self.is_recording:
            self.events.append((beat_position, wave_array))
            self.has_content = True

    def update(self, transport: Transport, audio_engine):
        """
        Call once per frame for the CURRENTLY ACTIVE mode's loop slot.
        Handles: (a) flipping armed->recording exactly on beat-1,
        (b) stopping a full 8-beat recording automatically,
        (c) replaying captured events at the right moment each loop cycle.
        """
        is_beat_one = transport.is_beat_one()

        # (a) Arm -> actually start recording on the next beat-1
        if self._armed and is_beat_one and not self.is_recording:
            self.events = []  # fresh recording, replaces whatever was here
            self.is_recording = True
            self._armed = False
            self._record_start_cycle = int(transport.elapsed() // transport.loop_duration)

        # (b) Auto-stop after exactly one full 8-beat cycle
        elif self.is_recording and is_beat_one:
            current_cycle = int(transport.elapsed() // transport.loop_duration)
            if current_cycle != self._record_start_cycle:
                self.is_recording = False

        # (c) Playback: each time we enter a new loop cycle, replay every
        # captured event at its stored beat position relative to cycle start.
        current_cycle = int(transport.elapsed() // transport.loop_duration)
        if self.has_content and not self.is_recording and current_cycle != self._last_played_cycle:
            self._last_played_cycle = current_cycle
            self._scheduled = list(self.events)  # events still due this cycle

        # Fire any scheduled events whose beat position we've now reached.
        if self.has_content and not self.is_recording and hasattr(self, "_scheduled"):
            pos = transport.loop_position_seconds() / transport.beat_duration
            still_pending = []
            for beat_pos, wave in self._scheduled:
                if pos >= beat_pos:
                    audio_engine.play_sound(wave)
                else:
                    still_pending.append((beat_pos, wave))
            self._scheduled = still_pending


class Looper:
    """Holds one LoopSlot per mode."""

    def __init__(self, mode_names):
        self.slots = {name: LoopSlot(name) for name in mode_names}

    def slot(self, mode_name) -> LoopSlot:
        return self.slots[mode_name]