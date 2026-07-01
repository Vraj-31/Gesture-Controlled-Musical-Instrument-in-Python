"""
audio_engine.py
----------------
PHASE 2 — Real-time audio playback engine.

WHY THIS EXISTS (rather than just calling sounddevice.play()):
  sounddevice's simple play() function explicitly CANNOT handle multiple
  overlapping sounds -- calling it again stops whatever was already
  playing. Since this project needs a chord strum and a drum hit (and later,
  multiple looped layers) to all sound at once, we need a persistent
  sd.OutputStream with our own mixing callback instead. This is the standard
  approach for any real-time multi-voice audio app in Python.

ARCHITECTURE:
  - AudioEngine opens ONE OutputStream when started, and keeps it open for
    the life of the program (opening/closing streams repeatedly adds latency
    and can click/pop).
  - play_sound(wave_array) adds a new "Voice" (a playhead into that array)
    to a list of currently-active voices.
  - The stream's callback runs in a separate high-priority audio thread
    (managed by PortAudio/sounddevice, not by us) every ~10-20ms, pulls the
    next chunk from every active voice, sums them, soft-clips, and writes
    the result to the output buffer.
  - Finished voices (played to the end of their array) are dropped automatically.

THREAD SAFETY NOTE:
  play_sound() is called from your MAIN thread (the webcam/CV loop), while
  the mixing happens in PortAudio's audio thread. We use a simple list with
  append/filter, which is safe enough for this use case in CPython, but if
  you ever see crashes under heavy concurrent triggering, switch self.voices
  to use a threading.Lock around mutations.

LATENCY:
  Total latency = OutputStream blocksize / sample_rate + a small fixed
  driver overhead. Default blocksize below targets ~10-20ms, which is
  fast enough to feel "instant" for a hand gesture -> sound use case.
  If you hear crackling/dropouts, RAISE blocksize (trades latency for
  stability). If playback feels laggy, LOWER it.
"""

import os
import tempfile
import threading
import wave

import numpy as np

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except (OSError, ModuleNotFoundError):
    # PortAudio native library missing. We still want sound_synth.py and
    # strum_detector.py to be importable/testable without audio hardware,
    # so we degrade gracefully here instead of crashing on import.
    SOUNDDEVICE_AVAILABLE = False

try:
    import winsound
    WINSOUND_AVAILABLE = os.name == "nt"
except ImportError:
    WINSOUND_AVAILABLE = False

SAMPLE_RATE = 44100
BLOCK_SIZE = 512  # ~11.6ms per block at 44100Hz -- good balance for this use case


class _Voice:
    """One currently-playing sound: a waveform array plus a read position."""
    __slots__ = (
        "samples", "pos", "group", "gain",
        "_fade_remaining", "_fade_total", "_fade_start_gain",
    )

    def __init__(self, samples, group=None):
        self.samples = samples
        self.pos = 0
        self.group = group
        self.gain = 1.0
        self._fade_remaining = None
        self._fade_total = 0
        self._fade_start_gain = 1.0

    def start_fade_out(self, duration_samples):
        if duration_samples <= 0:
            self.gain = 0.0
            self.pos = len(self.samples)
            return
        self._fade_remaining = duration_samples
        self._fade_total = duration_samples
        self._fade_start_gain = self.gain

    def get_block(self, n):
        """Returns the next n samples, zero-padded if near the end.
        Returns None once fully consumed (signals the engine to drop this voice)."""
        remaining = len(self.samples) - self.pos
        if remaining <= 0:
            return None
        chunk = self.samples[self.pos: self.pos + n].copy()
        self.pos += n
        actual_len = len(chunk)

        if self._fade_remaining is None:
            chunk *= self.gain
        else:
            fade_len = min(actual_len, self._fade_remaining)
            if fade_len > 0:
                start_level = self._fade_remaining / self._fade_total
                end_remaining = self._fade_remaining - fade_len
                end_level = end_remaining / self._fade_total
                fade_curve = np.linspace(start_level, end_level, fade_len, endpoint=False)
                chunk[:fade_len] *= self._fade_start_gain * fade_curve
                self._fade_remaining = end_remaining
            if fade_len < actual_len:
                chunk[fade_len:] = 0.0
            if self._fade_remaining <= 0:
                self.gain = 0.0
                self.pos = len(self.samples)

        if len(chunk) < n:
            chunk = np.pad(chunk, (0, n - len(chunk)))
        return chunk


class AudioEngine:
    def __init__(self, sample_rate=SAMPLE_RATE, block_size=BLOCK_SIZE, master_volume=0.9):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.master_volume = master_volume

        self._voices = []
        self._lock = threading.Lock()
        self._stream = None
        self._running = False
        self._using_fallback_audio = False
        self._fallback_files = []

    def start(self):
        """Opens the persistent output stream. Call once at program start."""
        if not SOUNDDEVICE_AVAILABLE:
            if WINSOUND_AVAILABLE:
                print("WARNING: sounddevice/PortAudio not available; using Windows fallback audio.")
                print("Fallback audio plays one sound at a time, but it will still produce sound.")
                self._running = True
                self._using_fallback_audio = True
            else:
                print("WARNING: sounddevice/PortAudio not available on this system.")
                print("Audio playback is disabled, but the rest of the app will still run.")
                print("(Hand tracking and strum detection work independently of audio.)")
            return

        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            channels=1,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()
        self._running = True

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
        self._running = False
        self._using_fallback_audio = False
        if WINSOUND_AVAILABLE:
            winsound.PlaySound(None, winsound.SND_PURGE)
        for path in self._fallback_files:
            try:
                os.remove(path)
            except OSError:
                pass
        self._fallback_files.clear()

    def play_sound(self, wave_array, group=None):
        """
        Triggers playback of a sound (numpy float32 array, mono, at this
        engine's sample_rate). Returns immediately -- playback happens in
        the background audio thread. Safe to call rapidly / multiple times
        in a row; sounds overlap rather than replacing each other.
        """
        if not self._running:
            return  # silently no-op if audio isn't available; CV/gestures still work
        if self._using_fallback_audio:
            self._play_sound_with_winsound(wave_array)
            return
        with self._lock:
            self._voices.append(_Voice(wave_array, group=group))

    def fade_group(self, group, duration_seconds):
        """Fade out currently-playing voices that belong to the given group."""
        duration_samples = int(duration_seconds * self.sample_rate)
        with self._lock:
            for voice in self._voices:
                if voice.group == group:
                    voice.start_fade_out(duration_samples)

    def _play_sound_with_winsound(self, wave_array):
        samples = np.asarray(wave_array, dtype=np.float32)
        samples = np.clip(samples, -1.0, 1.0)
        pcm = (samples * 32767).astype(np.int16)

        fd, path = tempfile.mkstemp(prefix="air_guitar_", suffix=".wav")
        os.close(fd)
        with wave.open(path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(pcm.tobytes())

        self._fallback_files.append(path)
        winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)

    def active_voice_count(self):
        with self._lock:
            return len(self._voices)

    def _callback(self, outdata, frames, time_info, status):
        """
        Called by PortAudio in its own audio thread roughly every
        block_size/sample_rate seconds. Must be fast and must not block
        (no file I/O, no large allocations if avoidable) or you'll hear
        crackling/dropouts.
        """
        if status:
            # e.g. underflow warnings -- printing here is fine since this
            # is meant for development; consider removing for a "shipped" build.
            print("Audio status:", status)

        mix = np.zeros(frames, dtype=np.float32)

        with self._lock:
            still_active = []
            for voice in self._voices:
                block = voice.get_block(frames)
                if block is not None:
                    mix += block
                    still_active.append(voice)
            self._voices = still_active

        mix *= self.master_volume
        np.clip(mix, -1.0, 1.0, out=mix)  # soft safety clip, avoids harsh digital distortion

        outdata[:, 0] = mix


# ---------------------------------------------------------------------------
# Self-test: verify the mixing logic works correctly using a fake stream
# (no real audio hardware needed). This tests the actual _Voice/_callback
# math, just without opening a real PortAudio stream.
# Run directly: python audio_engine.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Running audio_engine.py self-test (mixing logic only, no hardware)...\n")

    engine = AudioEngine()
    # Bypass start() (which needs real hardware) and test mixing directly.
    engine._running = True

    tone_a = (np.sin(2 * np.pi * 220 * np.linspace(0, 0.05, 2205, endpoint=False)) * 0.5).astype(np.float32)
    tone_b = (np.sin(2 * np.pi * 440 * np.linspace(0, 0.05, 2205, endpoint=False)) * 0.5).astype(np.float32)

    engine.play_sound(tone_a)
    engine.play_sound(tone_b)
    assert engine.active_voice_count() == 2, "expected 2 active voices after 2 play_sound calls"
    print(f"  OK  2 overlapping sounds queued (active_voice_count={engine.active_voice_count()})")

    # Simulate what PortAudio would do: repeatedly pull blocks until voices drain.
    fake_outdata = np.zeros((engine.block_size, 1), dtype=np.float32)
    blocks_pulled = 0
    max_blocks = 50  # safety limit
    while engine.active_voice_count() > 0 and blocks_pulled < max_blocks:
        engine._callback(fake_outdata, engine.block_size, None, None)
        blocks_pulled += 1
        assert not np.isnan(fake_outdata).any(), "NaN detected in mixed output!"
        assert np.max(np.abs(fake_outdata)) <= 1.0001, "clipping beyond +-1 detected!"

    print(f"  OK  both voices fully drained after {blocks_pulled} blocks, no NaN/clipping")
    assert engine.active_voice_count() == 0, "voices should be empty after draining"
    print(f"  OK  active_voice_count correctly returns to 0")

    engine.play_sound(tone_a, group="strings")
    engine.fade_group("strings", 0.01)
    engine._callback(fake_outdata, engine.block_size, None, None)
    engine._callback(fake_outdata, engine.block_size, None, None)
    assert engine.active_voice_count() == 0, "faded voice should be removed"
    print(f"  OK  grouped fade-out drains voices cleanly")

    print(f"\n  sounddevice/PortAudio available on this system: {SOUNDDEVICE_AVAILABLE}")
    if not SOUNDDEVICE_AVAILABLE:
        print("  (Mixing logic is verified correct above; real playback needs")
        print("   PortAudio installed, which this self-test doesn't require.)")

    print("\nSelf-test PASSED: mixing engine correctly overlaps and drains multiple voices.")
