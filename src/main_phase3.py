"""
main_phase3.py
---------------
Mode-switching instrument with metronome + looper + real air-drums.

KEYS:
  1 / 2 / 3   -> switch mode: Guitar / Drums / Strings
  m           -> toggle metronome on/off
  r           -> arm recording for the CURRENT mode's loop (starts on next beat 1)
  c           -> clear the current mode's loop
  q           -> quit

ONLY the active mode's gesture logic and overlay run/show each frame --
this is the "don't show everything at once" behavior you asked for.
"""

import math
import time
from pathlib import Path

import cv2

from hand_tracker import HandTracker
from fps_counter import FPSCounter
from audio_engine import AudioEngine
from transport import Transport
from looper import Looper

from strum_detector import StrumDetector
from chord_selector import select_chord, count_fingers_up
from drum_detector import DrumDetector, DRUM_ZONES
from strings_selector import (
    detect_root_note_index_from_xy,
    detect_chord_quality_from_xy,
    chord_frequencies,
    chord_name,
    NOTE_ZONES,
    QUALITY_ZONES,
)

from sound_synth import (
    load_wav_sample,
    synthesize_chord,
    synthesize_sustained_chord,
    CHORDS,
)

MODES = ["guitar", "drums", "strings"]
STRINGS_RETRIGGER_SECONDS = 1.55
STRINGS_CHANGE_DELAY_SECONDS = 0.5
STRINGS_FADE_SECONDS = 0.45
STRINGS_AUDIO_GROUP = "strings"
HAND_POINTER_LANDMARK = 8  # index fingertip feels natural for choosing screen zones
THUMB_TIP_LANDMARK = 4
INDEX_TIP_LANDMARK = 8
INDEX_MCP_LANDMARK = 5
GUITAR_PICK_JOIN_RATIO = 0.42
GUITAR_STRUM_VELOCITY_THRESHOLD = 0.8
DRUM_SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sound_previews"
DRUM_SAMPLE_FILES = {
    "kick": "kick.wav",
    "snare": "snare.wav",
    "hihat": "hihat.wav",
    "crash": "crash.wav",
    "ride": "ride.wav",
    "rack_tom_high": "rack_tom_high.wav",
    "rack_tom_mid": "rack_tom_mid.wav",
    "floor_tom": "floor_tom.wav",
}


def pregenerate_guitar_sounds():
    sounds = {}
    for name in CHORDS:
        sounds[(name, "DOWN")] = synthesize_chord(name, strum_direction="DOWN")
        sounds[(name, "UP")] = synthesize_chord(name, strum_direction="UP")
    return sounds


def pregenerate_drum_sounds():
    missing = [
        filename
        for filename in DRUM_SAMPLE_FILES.values()
        if not (DRUM_SAMPLE_DIR / filename).exists()
    ]
    if missing:
        needed = ", ".join(missing)
        raise FileNotFoundError(
            f"Missing drum sample(s) in {DRUM_SAMPLE_DIR}: {needed}"
        )

    return {
        zone: load_wav_sample(DRUM_SAMPLE_DIR / filename)
        for zone, filename in DRUM_SAMPLE_FILES.items()
    }


def _distance(a, b):
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def detect_guitar_pick(landmarks):
    thumb = landmarks[THUMB_TIP_LANDMARK]
    index = landmarks[INDEX_TIP_LANDMARK]
    wrist = landmarks[0]
    index_mcp = landmarks[INDEX_MCP_LANDMARK]

    hand_scale = max(_distance(wrist, index_mcp), 0.001)
    pinch_ratio = _distance(thumb, index) / hand_scale
    joined = pinch_ratio <= GUITAR_PICK_JOIN_RATIO

    pick_x = (thumb["x"] + index["x"]) * 0.5
    pick_y = (thumb["y"] + index["y"]) * 0.5
    return joined, pinch_ratio, pick_x, pick_y


def draw_guitar_pick_indicator(frame, right_hand, joined, pinch_ratio, pick_x, pick_y):
    h, w, _ = frame.shape
    thumb_px = right_hand["landmarks_px"][THUMB_TIP_LANDMARK]
    index_px = right_hand["landmarks_px"][INDEX_TIP_LANDMARK]
    pick_px = (int(pick_x * w), int(pick_y * h))
    color = (0, 255, 0) if joined else (80, 80, 255)
    status = "JOINED" if joined else "OPEN"

    cv2.line(frame, thumb_px, index_px, color, 3)
    cv2.circle(frame, thumb_px, 7, color, 2)
    cv2.circle(frame, index_px, 7, color, 2)
    cv2.circle(frame, pick_px, 10, color, 2)
    cv2.putText(frame, f"Pick: {status} ({pinch_ratio:.2f})",
                (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_strings_zones(frame, selected_note_idx=None, selected_quality=None):
    h, w, _ = frame.shape

    cv2.putText(frame, "NOTES", (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 220, 80), 2)
    for idx, (name, (x0, x1, y0, y1)) in enumerate(NOTE_ZONES.items()):
        pt1 = (int(x0 * w), int(y0 * h))
        pt2 = (int(x1 * w), int(y1 * h))
        selected = selected_note_idx == idx
        color = (255, 220, 80) if selected else (120, 120, 120)
        thickness = 3 if selected else 1
        cv2.rectangle(frame, pt1, pt2, color, thickness)
        label_pos = (pt1[0] + 10, pt1[1] + int((pt2[1] - pt1[1]) * 0.58))
        cv2.putText(frame, name, label_pos, cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, color, 2)

    cv2.putText(frame, "VARIATIONS", (int(0.52 * w), 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 220, 255), 2)
    for name, (x0, x1, y0, y1) in QUALITY_ZONES.items():
        pt1 = (int(x0 * w), int(y0 * h))
        pt2 = (int(x1 * w), int(y1 * h))
        selected = selected_quality == name
        color = (120, 220, 255) if selected else (120, 120, 120)
        thickness = 3 if selected else 1
        cv2.rectangle(frame, pt1, pt2, color, thickness)
        font_scale = 0.52 if len(name) > 7 else 0.6
        label_pos = (pt1[0] + 8, pt1[1] + int((pt2[1] - pt1[1]) * 0.55))
        cv2.putText(frame, name, label_pos, cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, 2)


def main():
    try:
        tracker = HandTracker(max_hands=2)
    except FileNotFoundError as e:
        print(str(e))
        return

    print("Pre-generating sounds...")
    try:
        guitar_sounds = pregenerate_guitar_sounds()
        drum_sounds = pregenerate_drum_sounds()
    except FileNotFoundError as e:
        print(str(e))
        tracker.close()
        return
    print("Done.")

    engine = AudioEngine()
    engine.start()

    transport = Transport(engine, bpm=90)
    transport.start()
    looper = Looper(MODES)

    strummer = StrumDetector(
        velocity_threshold=GUITAR_STRUM_VELOCITY_THRESHOLD,
        debounce_seconds=0.12,
    )
    drummer = DrumDetector()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        tracker.close(); engine.stop(); return

    fps_counter = FPSCounter()
    current_mode_idx = 0
    current_chord = "Em"  # guitar mode state
    last_strings_chord_key = None
    last_strings_trigger_time = 0.0
    pending_strings_chord_key = None
    pending_strings_started_time = 0.0

    print("\n1/2/3 = mode | m = metronome | r = record | c = clear loop | q = quit\n")

    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        hands_data = tracker.process(frame)
        frame = tracker.draw(frame, hands_data)
        left_hand = next((hd for hd in hands_data if hd["label"] == "Left"), None)
        right_hand = next((hd for hd in hands_data if hd["label"] == "Right"), None)

        transport.update()  # metronome click happens inside here
        mode = MODES[current_mode_idx]
        slot = looper.slot(mode)
        slot.update(transport, engine)  # plays back this mode's loop if it has content

        # ---------------- GUITAR MODE ----------------
        if mode == "guitar":
            if left_hand is not None:
                current_chord = select_chord(left_hand["landmarks"])
                cv2.putText(frame, f"Chord: {current_chord} (fingers={count_fingers_up(left_hand['landmarks'])})",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            if right_hand is not None:
                joined, pinch_ratio, pick_x, pick_y = detect_guitar_pick(right_hand["landmarks"])
                draw_guitar_pick_indicator(frame, right_hand, joined, pinch_ratio, pick_x, pick_y)

                if joined:
                    event = strummer.update(pick_y)
                    if event is not None:
                        wave = guitar_sounds[(current_chord, event.direction)]
                        engine.play_sound(wave)
                        beat_pos = (transport.loop_position_seconds() / transport.beat_duration)
                        slot.capture_event(beat_pos, wave)
                else:
                    strummer.reset()

                display_velocity = strummer.last_velocity if joined else 0.0
                cv2.putText(frame, f"Pick vel: {display_velocity:+.2f}",
                            (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                strummer.reset()

        # ---------------- DRUMS MODE ----------------
        elif mode == "drums":
            left_points = (
                [(lm["x"], lm["y"]) for lm in left_hand["landmarks"]]
                if left_hand else None
            )
            right_points = (
                [(lm["x"], lm["y"]) for lm in right_hand["landmarks"]]
                if right_hand else None
            )
            hits = drummer.update(left_points, right_points)
            for zone in hits:
                if zone is None:
                    continue
                wave = drum_sounds[zone]
                engine.play_sound(wave)
                beat_pos = (transport.loop_position_seconds() / transport.beat_duration)
                slot.capture_event(beat_pos, wave)
                print(f"HIT: {zone}")

            for name, (x0, x1, y0, y1) in DRUM_ZONES.items():
                pt1 = (int(x0 * w), int(y0 * h))
                pt2 = (int(x1 * w), int(y1 * h))
                cv2.rectangle(frame, pt1, pt2, (0, 200, 255), 2)
                cv2.putText(frame, name, (pt1[0] + 5, pt1[1] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        # ---------------- STRINGS MODE ----------------
        elif mode == "strings":
            root_idx = None
            quality = None

            if left_hand is not None:
                pointer = left_hand["landmarks"][HAND_POINTER_LANDMARK]
                root_idx = detect_root_note_index_from_xy(pointer["x"], pointer["y"])

            if right_hand is not None:
                pointer = right_hand["landmarks"][HAND_POINTER_LANDMARK]
                quality = detect_chord_quality_from_xy(pointer["x"], pointer["y"])

            draw_strings_zones(frame, root_idx, quality)

            if left_hand is not None and right_hand is not None:
                if root_idx is not None and quality is not None:
                    freqs = chord_frequencies(root_idx, quality)
                    current_strings_chord = chord_name(root_idx, quality)
                    cv2.putText(frame, current_strings_chord,
                                (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 0), 2)

                    now = time.time()
                    chord_key = (root_idx, quality)
                    should_trigger = False

                    if chord_key == last_strings_chord_key:
                        pending_strings_chord_key = None
                        pending_strings_started_time = 0.0
                        should_trigger = (
                            now - last_strings_trigger_time
                        ) >= STRINGS_RETRIGGER_SECONDS
                    elif chord_key != pending_strings_chord_key:
                        pending_strings_chord_key = chord_key
                        pending_strings_started_time = now
                    elif (now - pending_strings_started_time) >= STRINGS_CHANGE_DELAY_SECONDS:
                        last_strings_chord_key = chord_key
                        pending_strings_chord_key = None
                        pending_strings_started_time = 0.0
                        should_trigger = True

                    if should_trigger:
                        last_strings_trigger_time = now
                        wave = synthesize_sustained_chord(freqs)
                        engine.fade_group(STRINGS_AUDIO_GROUP, STRINGS_FADE_SECONDS)
                        engine.play_sound(wave, group=STRINGS_AUDIO_GROUP)
                        print(f"STRINGS: {current_strings_chord}")
                        beat_pos = (transport.loop_position_seconds() / transport.beat_duration)
                        slot.capture_event(beat_pos, wave)
                else:
                    if last_strings_chord_key is not None or pending_strings_chord_key is not None:
                        engine.fade_group(STRINGS_AUDIO_GROUP, STRINGS_FADE_SECONDS)
                    last_strings_chord_key = None
                    pending_strings_chord_key = None
                    pending_strings_started_time = 0.0
                    cv2.putText(frame, "Select note + variation",
                                (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
            else:
                if last_strings_chord_key is not None or pending_strings_chord_key is not None:
                    engine.fade_group(STRINGS_AUDIO_GROUP, STRINGS_FADE_SECONDS)
                last_strings_chord_key = None
                pending_strings_chord_key = None
                pending_strings_started_time = 0.0
                cv2.putText(frame, "Show both hands",
                            (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)

        # ---------------- shared UI ----------------
        cv2.putText(frame, f"MODE: {mode.upper()}  [1=guitar 2=drums 3=strings]",
                    (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        metro_status = "ON" if transport.metronome_on else "off"
        cv2.putText(frame, f"Metronome: {metro_status} (m)  Beat: {transport.current_beat_index()+1}/8",
                    (10, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
        rec_status = "RECORDING" if slot.is_recording else ("ARMED" if slot._armed else
                     ("LOOP ACTIVE" if slot.has_content else "empty"))
        cv2.putText(frame, f"Loop[{mode}]: {rec_status} (r=record, c=clear)",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        cv2.imshow("Air Instrument - Phase 3", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            current_mode_idx = 0
        elif key == ord('2'):
            current_mode_idx = 1
        elif key == ord('3'):
            current_mode_idx = 2
        elif key == ord('m'):
            transport.toggle_metronome()
        elif key == ord('r'):
            slot.arm_record()
        elif key == ord('c'):
            slot.clear()

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()
    engine.stop()


if __name__ == "__main__":
    main()
