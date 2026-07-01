# Air Guitar — Phase 1 + Phase 2

Gesture-controlled virtual instrument project.
- **Phase 1**: real-time webcam hand tracking with reliable left/right hand labeling.
- **Phase 2**: a playable two-handed instrument on top of that — left hand
  picks a chord, right hand strums it, and a drum zone triggers a kick —
  all with real, audible, overlapping sound.

## What this does

Opens your webcam, detects up to 2 hands in real time, draws the hand
skeleton on screen, and labels each hand "Left" or "Right" with a confidence
score. Also shows FPS and live wrist coordinates — the raw signal that
Phase 2 (strum detection) tracks over time.

## What Phase 2 adds

A genuinely playable instrument, end to end:

- **Left hand = chord picker.** Show 0–4 fingers to select one of 5 open
  chords (Em, G, C, D, Am).
- **Right hand = strummer.** Move your right hand quickly up or down to
  "strum" — this triggers the currently-selected chord, with a slightly
  different sound for down-strums vs up-strums (real strums sweep the
  strings in a direction, and this mimics that).
- **Drum zone.** Move your right hand into the red box (bottom-right of the
  frame) to trigger a kick drum, independent of strumming.
- **Real overlapping audio.** A chord and a drum hit can sound at the same
  time without cutting each other off — this needed a proper small audio
  mixing engine (see `audio_engine.py`), not just simple "play a sound" calls.
- **No external audio files needed.** All chords and drum sounds are
  synthesized procedurally in Python (`sound_synth.py`) so Phase 2 runs
  immediately — you can swap in real recorded samples later without
  changing any other file.

## Requirements

- Python 3.9–3.12
- A webcam
- Speakers/headphones (for Phase 2)
- ~10MB free disk space (for the hand-tracking model file)
- **Linux only**: the native PortAudio library, since pip's `sounddevice`
  package doesn't include it on Linux (it usually does on Windows/Mac).
  `setup.py` checks for this and tells you the exact fix command if it's
  missing — e.g. `sudo apt-get install libportaudio2` on Ubuntu/Debian.

## Setup (one-time)

```bash
cd src
pip install mediapipe opencv-python numpy scipy sounddevice
python setup.py
```

`setup.py` does two things:
1. Downloads `hand_landmarker.task` (the ML model file) into `src/`.
2. Checks that `sounddevice` can actually talk to your speakers (PortAudio),
   and tells you exactly how to fix it if not — this is the #1 thing that
   silently breaks on a fresh Linux install.

If your network blocks the model download, the script prints a direct URL
you can open in a browser instead — just save the file as
`hand_landmarker.task` inside `src/`.

## Run it

**Phase 1** (hand tracking only, no sound):
```bash
python main.py
```

**Phase 2** (full instrument — chords, strumming, drums, real audio):
```bash
python main_phase2.py
```

A window opens showing your webcam feed, mirrored (like a selfie cam), with
hand skeletons overlaid. Press **q** to quit either one.

### How to play Phase 2

1. Hold up your **left hand**, palm toward the camera. The number of
   extended fingers (0–4) picks the chord — watch the on-screen label.
2. Hold up your **right hand** and move it quickly **down**, then quickly
   **up**, like an actual strum. Each fast motion plays the currently
   selected chord.
3. Move your **right hand** into the **red box** in the bottom-right corner
   to trigger a kick drum.
4. Try changing chords with your left hand while continuing to strum with
   your right — that's the two-handed "instrument" feel this phase is
   building toward.

## Project structure

```
src/
  hand_tracker.py      # Core MediaPipe wrapper — detection only, no gesture logic
  fps_counter.py        # Performance monitoring utility
  main.py                # Phase 1 entry point: webcam loop + debug overlay
  setup.py               # One-time model downloader + audio backend check

  strum_detector.py      # Phase 2: velocity-based strum gesture detection (pure logic)
  chord_selector.py      # Phase 2: left-hand finger count -> chord name (pure logic)
  sound_synth.py         # Phase 2: procedural synthesis of chords + drum sounds
  audio_engine.py        # Phase 2: real-time mixing engine for overlapping playback
  main_phase2.py          # Phase 2 entry point: wires everything together

  hand_landmarker.task   # (created by setup.py, ~10MB, not committed to git)
```

Every Phase 2 logic module (`strum_detector.py`, `chord_selector.py`,
`sound_synth.py`, `audio_engine.py`) has a built-in self-test you can run
directly, with zero hardware required:

```bash
python strum_detector.py    # tests strum detection against synthetic motion
python chord_selector.py    # tests finger counting against synthetic landmarks
python sound_synth.py       # verifies every synthesized waveform is valid audio
python audio_engine.py      # verifies the mixing logic handles overlapping sounds
```

Useful for confirming things still work after you tune any thresholds.

## Why these specific design choices

**Why `mediapipe.tasks` instead of the older `mp.solutions.hands`?**
Recent MediaPipe versions (0.10.x) removed the old `solutions` API entirely.
The current `tasks` API is also what the JavaScript/web version uses
(`@mediapipe/tasks-vision`), so the gesture-detection logic you build next
(strum detection, chord shapes) will port to a browser version with minimal
changes later, if you go that route.

**Why flip the frame horizontally?**
MediaPipe's left/right hand labels assume a mirrored/selfie-style camera.
Flipping the frame in `main.py` before detection makes the "Left"/"Right"
label match the user's actual hand, and feels natural on screen (move your
hand right, see it move right) — like looking in a mirror.

**Why `running_mode=VIDEO` instead of `IMAGE` or `LIVE_STREAM`?**
`VIDEO` mode processes frames synchronously with a timestamp, which is
simplest for a webcam loop. `LIVE_STREAM` mode is async/callback-based and
adds complexity we don't need yet — worth revisiting only if Phase 2+
performance needs it.

**Why a persistent `OutputStream` instead of just calling `sounddevice.play()`?**
`sounddevice.play()` explicitly cannot handle multiple overlapping sounds —
calling it again stops whatever was already playing. Since a chord strum and
a drum hit need to sound at once (and Phase 4's looper will need several
layers playing simultaneously), `audio_engine.py` opens one persistent
stream with its own mixing callback that sums all currently-playing sounds
each audio block. This is the standard pattern for any real-time multi-voice
audio app in Python, and it's the same architecture the looper will extend.

**Why synthesize sounds instead of using recorded samples?**
Zero external assets needed — Phase 2 runs immediately after `pip install`.
`sound_synth.py` has a `load_wav_sample()` function ready for when you want
to swap in real recorded guitar/drum samples; it returns audio in the exact
same format the synthesized sounds use, so it's a drop-in replacement.

**Why finger-count for chord selection instead of finger shapes?**
Counting extended fingers (0–4 → 5 chords) is the simplest gesture that's
genuinely easy to perform reliably and gets a real two-handed instrument
working today. `chord_selector.py` is intentionally isolated so Phase 3 can
replace it with proper per-finger chord shapes (e.g. specific finger
combinations, not just a count) without touching any other file.

## Troubleshooting

**"Could not open webcam"**
Another app may be using the camera, or your OS hasn't granted camera
permission to your terminal/IDE. Check System Settings → Privacy → Camera
(macOS) or just close other video apps (Zoom, Teams, browser tabs using the
camera).

**Low FPS (<15)**
Try lowering resolution further in `main.py` (e.g. 480x360), or reduce
`max_hands` to 1 while testing single-hand logic.

**Hands not detected reliably**
Improve lighting — MediaPipe's palm detector is lighting-sensitive. Avoid
strong backlight (window behind you). Keep hands fully in frame.

**Left/Right labels seem swapped**
This would only happen with an unusual camera setup. See the note in
`hand_tracker.py` above `_fix_label`-equivalent comments — the fix is a
one-line swap if you ever need it.

**No sound at all (Phase 2)**
Run `python setup.py` again and check the `[2/2]` audio backend section —
it will tell you directly if PortAudio is missing and how to install it.
On Linux: `sudo apt-get install libportaudio2`. On macOS: `brew install
portaudio`. If that section says OK but you still hear nothing, check your
system's output device/volume — `setup.py` prints which device it found as
default.

**Strums not triggering, or triggering too easily**
Watch the "R-wrist vel" number on screen while you physically strum. If real
strums show velocity well below the threshold, lower
`StrumDetector(velocity_threshold=...)` in `main_phase2.py` (default 1.8).
If small movements or jitter are falsely triggering strums, raise it.

**Crackling / choppy audio**
Raise `BLOCK_SIZE` in `audio_engine.py` (default 512) — this trades a little
latency for stability. If your machine is also under load from the CV
pipeline, lowering webcam resolution can help free up CPU for audio too.

**Chord sounds feel delayed after strumming**
This is more likely a strum-detection timing issue than audio latency —
check the debounce/threshold tuning above first. If the audio itself feels
laggy, lower `BLOCK_SIZE` in `audio_engine.py` (trades stability for speed).

## What's next (Phase 3 preview)

Phase 3 upgrades `chord_selector.py` from simple finger-counting to real
finger-shape recognition (specific combinations, not just a count) for more
expressive control, and likely splits the drum trigger so it doesn't share
the strum hand. Phase 4 adds the looper: a fixed-tempo metronome and the
ability to record a layer (chords or drums) that loops continuously while
you record more layers on top — building directly on the mixing engine
`audio_engine.py` already provides.
