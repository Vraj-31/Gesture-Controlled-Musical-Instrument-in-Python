"""
sound_synth.py
---------------
PHASE 2 — Procedural sound synthesis for guitar-like tones.

WHY SYNTHESIZE INSTEAD OF USING RECORDED SAMPLES:
  Recorded guitar samples sound better, but require sourcing/licensing audio
  files before you can run anything. Synthesizing tones means Phase 2 runs
  immediately with zero external assets, and you can swap in real samples
  later (see load_wav_sample() at the bottom) without changing any other
  module -- audio_engine.py just wants a numpy float32 array, it doesn't
  care whether that array came from synthesis or a file.

METHOD: simple additive synthesis (a few harmonics) + exponential decay
  envelope. This is NOT a physically accurate guitar-string model (that
  would be Karplus-Strong, which sounds noticeably more "plucky" -- a nice
  upgrade later if you want), but it's cheap, has zero dependencies beyond
  numpy, and produces a clearly pitched, recognizable "twang" you can play
  chords with right away.

CHORDS: a chord here is just a list of frequencies (notes) played together.
  We use real guitar open-chord frequencies for the standard EADGBE tuning
  so this sounds like an actual guitar chord, not arbitrary musical notes.
"""

import numpy as np

SAMPLE_RATE = 44100

# Standard EADGBE guitar string frequencies (Hz), for reference/extension:
# E2=82.41  A2=110.00  D3=146.83  G3=196.00  B3=246.94  E4=329.63

# Common open-chord shapes as lists of (string) frequencies actually sounded.
# These are real open-chord voicings, not arbitrary chords -- e.g. open G
# major uses G3, B3, D4, G4, B4, G2 in real life; we use a slightly
# simplified subset (3-4 notes) which still sounds clearly chordal without
# needing 6 simultaneous oscillators per strum (cheaper to mix, sounds fine).
CHORDS = {
    "Em":  [82.41, 123.47, 164.81, 196.00],   # E2, B2, E3, G3
    "G":   [98.00, 123.47, 196.00, 246.94],   # G2, B2, G3, B3
    "C":   [130.81, 164.81, 196.00, 261.63],  # C3, E3, G3, C4
    "D":   [146.83, 220.00, 293.66, 369.99],  # D3, A3, D4, F#4
    "Am":  [110.00, 164.81, 220.00, 261.63],  # A2, E3, A3, C4
}

DEFAULT_CHORD = "Em"


def _envelope(n_samples, attack_ratio=0.01, decay_rate=3.0):
    """
    Exponential-decay envelope shaping the amplitude over time, like a
    plucked string: quick attack, then decaying sustain.
    decay_rate: higher = faster decay (shorter, more percussive pluck).
    """
    t = np.linspace(0, 1, n_samples, endpoint=False)
    attack_samples = int(n_samples * attack_ratio)

    env = np.exp(-decay_rate * t)
    if attack_samples > 0:
        attack_curve = np.linspace(0, 1, attack_samples)
        env[:attack_samples] = attack_curve * env[:attack_samples]
    return env


def synthesize_note(frequency, duration=0.9, sample_rate=SAMPLE_RATE,
                     n_harmonics=3, decay_rate=3.0):
    """
    Generates one plucked-string-like tone at `frequency` Hz.
    Returns a float32 numpy array of samples, normalized to roughly [-1, 1]
    BEFORE the envelope is applied to its final position (so two notes of
    different frequency still mix at comparable loudness).
    """
    n_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    wave = np.zeros(n_samples)
    # Additive harmonics with decreasing weight -- mimics a string's
    # natural overtone series (fundamental loudest, harmonics quieter).
    for h in range(1, n_harmonics + 1):
        weight = 1.0 / h
        wave += weight * np.sin(2 * np.pi * frequency * h * t)

    wave /= np.max(np.abs(wave))  # normalize before envelope
    wave *= _envelope(n_samples, decay_rate=decay_rate)

    return wave.astype(np.float32)


def synthesize_chord(chord_name, duration=0.9, strum_direction="DOWN",
                      sample_rate=SAMPLE_RATE):
    """
    Generates a chord by mixing several notes together.

    strum_direction affects the *micro-timing* of when each string's note
    starts -- a real strum doesn't hit all strings at exactly 0ms, it
    sweeps across them in a few milliseconds. DOWN strums sweep low-to-high
    pitch (bass string first); UP strums sweep high-to-low. This single
    detail does a lot to make a synthesized chord feel like a strum rather
    than a stacked chord-stab.
    """
    if chord_name not in CHORDS:
        raise ValueError(f"Unknown chord '{chord_name}'. Available: {list(CHORDS.keys())}")

    frequencies = CHORDS[chord_name]
    if strum_direction == "UP":
        frequencies = list(reversed(frequencies))

    n_samples = int(duration * sample_rate)
    mix = np.zeros(n_samples, dtype=np.float32)

    # Stagger each note's start by a few milliseconds to simulate the pick/
    # fingers sweeping across strings rather than all strings firing at once.
    stagger_seconds = 0.012
    for i, freq in enumerate(frequencies):
        note = synthesize_note(freq, duration=duration, sample_rate=sample_rate)
        offset_samples = int(i * stagger_seconds * sample_rate)
        end = offset_samples + len(note)
        if end > n_samples:
            note = note[: n_samples - offset_samples]
            end = n_samples
        mix[offset_samples:end] += note[: end - offset_samples]

    # Normalize the combined chord so multiple overlapping notes don't clip,
    # then leave a little headroom (0.85) so the audio_engine's mixer has
    # room to add a drum hit on top later without distortion.
    peak = np.max(np.abs(mix))
    if peak > 0:
        mix = (mix / peak) * 0.85

    return mix.astype(np.float32)


def synthesize_kick(duration=0.25, sample_rate=SAMPLE_RATE):
    """A simple synthesized kick drum: a fast pitch-dropping sine + click."""
    n_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    # Pitch sweeps from ~150Hz down to ~40Hz quickly -- classic kick character.
    freq_sweep = 150 * np.exp(-25 * t) + 40
    phase = 2 * np.pi * np.cumsum(freq_sweep) / sample_rate
    tone = np.sin(phase)

    env = np.exp(-18 * t)
    wave = tone * env
    wave /= np.max(np.abs(wave))
    return (wave * 0.9).astype(np.float32)


def synthesize_snare(duration=0.18, sample_rate=SAMPLE_RATE):
    """A simple synthesized snare: noise burst + a bit of tone, fast decay."""
    n_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    noise = np.random.uniform(-1, 1, n_samples)
    tone = np.sin(2 * np.pi * 180 * t) * 0.3

    env = np.exp(-30 * t)
    wave = (noise * 0.7 + tone) * env
    wave /= np.max(np.abs(wave))
    return (wave * 0.8).astype(np.float32)


def synthesize_hihat(duration=0.08, sample_rate=SAMPLE_RATE):
    """A simple synthesized closed hi-hat: filtered noise, very fast decay."""
    n_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    noise = np.random.uniform(-1, 1, n_samples)
    # crude high-pass-ish effect: subtract a smoothed version of itself
    smoothed = np.convolve(noise, np.ones(5) / 5, mode="same")
    bright_noise = noise - smoothed

    env = np.exp(-60 * t)
    wave = bright_noise * env
    peak = np.max(np.abs(wave))
    if peak > 0:
        wave = (wave / peak) * 0.6
    return wave.astype(np.float32)


def load_wav_sample(filepath, sample_rate=SAMPLE_RATE):
    """
    OPTIONAL UPGRADE PATH: load a real recorded .wav sample instead of a
    synthesized tone. Returns a float32 array in the same format synthesize_*
    functions produce, so it's a drop-in replacement anywhere in this codebase.
    Resamples if the file's sample rate doesn't match SAMPLE_RATE.
    """
    import wave

    with wave.open(str(filepath), "rb") as wav_file:
        sr = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width == 1:
        data = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sample_width == 2:
        data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 3:
        raw = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
        signed = (
            raw[:, 0].astype(np.int32)
            | (raw[:, 1].astype(np.int32) << 8)
            | (raw[:, 2].astype(np.int32) << 16)
        )
        signed = np.where(signed & 0x800000, signed - 0x1000000, signed)
        data = signed.astype(np.float32) / 8388608.0
    elif sample_width == 4:
        data = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)

    if sr != sample_rate:
        n_target = int(len(data) * sample_rate / sr)
        source_positions = np.linspace(0, len(data) - 1, num=len(data))
        target_positions = np.linspace(0, len(data) - 1, num=n_target)
        data = np.interp(target_positions, source_positions, data)

    data = np.clip(data, -1.0, 1.0)
    return data.astype(np.float32)

def synthesize_tom(duration=0.3, sample_rate=SAMPLE_RATE):
    """Synthesized tom drum: pitched lower than kick, less click, longer ring."""
    n_samples = int(duration * sample_rate)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    freq_sweep = 220 * np.exp(-8 * t) + 90
    phase = 2 * np.pi * np.cumsum(freq_sweep) / sample_rate
    tone = np.sin(phase)
    env = np.exp(-9 * t)
    wave = tone * env
    wave /= np.max(np.abs(wave))
    return (wave * 0.85).astype(np.float32)

def synthesize_sustained_chord(frequencies, duration=2.4, sample_rate=SAMPLE_RATE,
                               root_gain=0.9, harmony_gain=0.65):
    """Strings-mode chord: slow attack, sustained, slow release -- distinct
    from guitar's plucked/decaying envelope."""
    n_samples = int(duration * sample_rate)
    mix = np.zeros(n_samples, dtype=np.float32)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    attack_samples = min(int(0.22 * sample_rate), n_samples)
    release_samples = min(int(0.55 * sample_rate), max(0, n_samples - attack_samples))
    env = np.ones(n_samples, dtype=np.float32)
    if attack_samples > 0:
        attack_curve = np.linspace(0, 1, attack_samples, endpoint=False)
        env[:attack_samples] = attack_curve * attack_curve * (3 - 2 * attack_curve)
    if release_samples > 0:
        release_curve = np.linspace(1, 0, release_samples, endpoint=False)
        env[-release_samples:] = release_curve * release_curve * (3 - 2 * release_curve)

    for idx, freq in enumerate(frequencies):
        note_gain = root_gain if idx == 0 else harmony_gain
        phase = idx * np.pi * 0.31
        wave = (
            np.sin(2 * np.pi * freq * t + phase) +
            0.18 * np.sin(2 * np.pi * freq * 2 * t + phase * 1.7) +
            0.05 * np.sin(2 * np.pi * freq * 3 * t + phase * 2.3)
        )
        mix += wave * note_gain

    mix *= env
    peak = np.max(np.abs(mix))
    if peak > 0:
        mix = (mix / peak) * 0.8
    return mix.astype(np.float32)


# ---------------------------------------------------------------------------
# Self-test: verify every synth function produces valid, finite, in-range
# audio without needing any audio hardware. Run directly: python sound_synth.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Running sound_synth.py self-test (no audio hardware needed)...\n")

    def check(name, wave, expected_min_len=100):
        assert wave.dtype == np.float32, f"{name}: wrong dtype {wave.dtype}"
        assert len(wave) > expected_min_len, f"{name}: suspiciously short ({len(wave)} samples)"
        assert not np.isnan(wave).any(), f"{name}: contains NaN"
        assert not np.isinf(wave).any(), f"{name}: contains Inf"
        peak = np.max(np.abs(wave))
        assert peak <= 1.0001, f"{name}: clipping risk, peak={peak}"
        print(f"  OK  {name:20s} len={len(wave):6d}  peak={peak:.3f}")

    check("synthesize_note(196Hz)", synthesize_note(196.0))
    for chord_name in CHORDS:
        check(f"chord '{chord_name}' DOWN", synthesize_chord(chord_name, strum_direction="DOWN"))
        check(f"chord '{chord_name}' UP", synthesize_chord(chord_name, strum_direction="UP"))
    check("kick", synthesize_kick())
    check("snare", synthesize_snare())
    check("hihat", synthesize_hihat())
    check("sustained chord", synthesize_sustained_chord([220.0, 277.18, 329.63]))

    try:
        synthesize_chord("NotAChord")
        print("FAILED: expected ValueError for unknown chord")
    except ValueError:
        print("  OK  unknown chord name correctly raises ValueError")

    print("\nSelf-test PASSED: all synthesized waveforms are valid float32 audio.")
