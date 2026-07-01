# 🎸 Air Instrument

A gesture-controlled virtual musical instrument that uses your webcam to track hand movements in real time. Play guitar, drums, and string instruments in the air without any physical hardware, powered by a custom audio engine that supports seamless, overlapping sound playback.

---

# Overview

Air Instrument transforms your hands into musical controllers using computer vision and real-time gesture recognition. The application detects up to **two hands simultaneously** through your webcam and converts different gestures into musical performances across multiple instrument modes.

Instead of relying on traditional input devices, the system interprets:

- Finger counting
- Pinching gestures
- Hand movement velocity
- Spatial pointing

to provide an intuitive and interactive virtual music experience.

---

# Features

## 🎼 Multiple Instrument Modes

Switch instantly between three different instruments using your keyboard.

| Key | Mode |
|------|------|
| **1** | Guitar |
| **2** | Drums |
| **3** | Strings |

Only the active instrument's gesture logic and visual overlays are processed, keeping the interface clean and improving performance.

---

## 🎸 Guitar Mode

Play guitar entirely in the air.

### Left Hand

- Hold your palm toward the camera.
- Raise **0–4 fingers** to select different chords.

### Right Hand

- Join your **thumb and index finger** to create a pick gesture.
- Move your hand **down** and **up** to strum.
- Open your fingers to disengage the pick.

This greatly reduces accidental strumming while moving your hand.

---

## 🥁 Drums Mode

Play a virtual drum kit using hand movement.

Move either hand into predefined screen regions to trigger different drum sounds:

- Kick
- Snare
- Hi-Hat
- Crash
- Ride
- High Tom
- Mid Tom
- Floor Tom

Each zone detects quick strikes for a natural drumming experience.

---

## 🎻 Strings Mode

Play sustained orchestral string chords.

### Left Hand

Point your index finger at the **Root Note** column.

Example:

- C
- D
- E
- F
- G
- A
- B

### Right Hand

Point your index finger at the **Chord Variation** column.

Examples:

- Major
- Minor
- Sus2
- Sus4
- Seventh

Holding both selections generates a sustained chord.

Changing either hand causes the current chord to smoothly crossfade into the newly selected chord.

---

## 🎵 Advanced Audio Engine

Unlike traditional playback methods, Air Instrument includes a custom real-time audio engine built with **sounddevice**.

Features include:

- Multiple overlapping sounds
- Persistent audio stream
- Smooth fade-outs
- Audio grouping
- Sustained notes
- Crossfading between chords
- Low-latency playback

---

# Requirements

- Python **3.9 – 3.12**
- Webcam
- Speakers or headphones
- Approximately **10 MB** free disk space (MediaPipe model)

### Linux Users

Install the native PortAudio library before running the project:

```bash
sudo apt-get install libportaudio2
```

Windows and macOS generally include the required PortAudio binaries through the `sounddevice` package.

---

# Installation

## 1. Install Python Dependencies

```bash
pip install mediapipe opencv-python numpy scipy sounddevice
```

---

## 2. Download the MediaPipe Hand Tracking Model

Navigate to the `src/` directory and run:

```bash
curl -o hand_landmarker.task -L https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

---

## 3. Add Drum Samples

Place the required `.wav` drum samples inside:

```text
sound_previews/
```

Example:

```text
sound_previews/
├── kick.wav
├── snare.wav
├── hihat.wav
├── crash.wav
├── ride.wav
├── tom1.wav
├── tom2.wav
└── tom3.wav
```

This folder should be located **one directory above** the main Python script.

---

# Running the Project

Execute:

```bash
python main_phase3.py
```

A webcam window will open with:

- mirrored camera feed
- detected hand landmarks
- gesture overlays
- instrument interface

---

# Keyboard Controls

| Key | Action |
|------|--------|
| **1** | Guitar Mode |
| **2** | Drums Mode |
| **3** | Strings Mode |
| **Q** | Quit Application |

---

# How to Play

## 🎸 Guitar Mode

1. Raise your **left hand**.
2. Hold up **0–4 fingers** to choose a chord.
3. Pinch your **right thumb** and **index finger**.
4. The on-screen indicator changes to **JOINED**.
5. Move your hand down and then up to strum.
6. Release the pinch to stop playing.

---

## 🥁 Drums Mode

1. Watch the yellow drum zones.
2. Move either hand quickly into a zone.
3. Each zone triggers a different drum sound.

---

## 🎻 Strings Mode

1. Point your **left index finger** at a root note.
2. Point your **right index finger** at a chord variation.
3. Hold both selections.
4. Move between zones to smoothly transition between sustained chords.

---

# Project Structure

```text
src/
│
├── hand_tracker.py        # MediaPipe wrapper and gesture detection
├── fps_counter.py         # FPS monitoring utility
│
├── main_phase3.py         # Main application
├── audio_engine.py        # Real-time audio mixer
├── sound_synth.py         # Procedural audio synthesis
│
├── strum_detector.py      # Velocity-based guitar strumming
├── chord_selector.py      # Finger count → chord mapping
├── drum_detector.py       # Drum zone detection
├── strings_selector.py    # Strings note selection
│
└── hand_landmarker.task   # MediaPipe model
```

---

# Technical Design

## Pinch-to-Pick Gesture

The guitar only activates when the thumb and index finger are pinched together.

This explicit "pick" gesture prevents accidental strumming while moving your hand around.

---

## Spatial Selection for Strings

Finger counting is unsuitable for sustained instruments because slight tracking errors can rapidly change notes.

Instead, large spatial regions are used for note selection, producing much more stable and deliberate chord transitions.

---

## Procedural Audio Synthesis

The guitar and string instruments are generated procedurally inside:

```text
sound_synth.py
```

No external audio assets are required for these instruments.

Only drum sounds rely on prerecorded `.wav` samples, as synthesizing realistic cymbals in real time is computationally expensive.

---

## Persistent Audio Stream

Using `sounddevice.play()` repeatedly interrupts previous sounds.

Instead, the application maintains a single persistent output stream with a custom mixing callback that:

- mixes active sounds
- supports overlapping playback
- performs fade-outs
- enables sustained instruments
- reduces audio artifacts

---

# Troubleshooting

## Webcam Cannot Be Opened

Another application may already be using your camera.

Close applications such as:

- Zoom
- Microsoft Teams
- Google Meet
- Browser tabs using the webcam

---

## Pinch Gesture Doesn't Trigger

Ensure that your thumb tip fully touches your index fingertip.

If detection is still difficult, increase:

```python
GUITAR_PICK_JOIN_RATIO
```

A slightly higher value (for example **0.5**) makes pinch detection more forgiving.

---

## Strings Flicker Between Notes

- Keep both hands visible.
- Hold your fingers steady inside the selection boxes.
- Avoid hovering near the borders of the regions.

---

## Crackling or Choppy Audio

Increase the audio block size in:

```text
audio_engine.py
```

Example:

```python
BLOCK_SIZE = 512
```

A larger block size increases stability at the cost of slightly higher latency.

Reducing webcam resolution can also improve performance on slower systems.

---

# Future Roadmap

## 🎙️ Integrated Looper

A multi-track looping system allowing users to:

- Record drum patterns
- Layer sustained strings
- Solo using the guitar
- Build complete performances in real time

---

## ⏱️ Metronome & Transport Controls

Future versions will include:

- Visual metronome
- Audible click track
- Tempo adjustment
- Beat synchronization
- Quantized loop recording

---

# Technologies Used

- Python
- OpenCV
- MediaPipe Tasks API
- NumPy
- SciPy
- sounddevice
- PortAudio

---

# License

This project is intended for educational and research purposes.

---

# Author

Developed as a real-time gesture-controlled virtual musical instrument using computer vision and procedural audio synthesis.