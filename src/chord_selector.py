"""
chord_selector.py
-------------------
PHASE 2 (simple v1) — Maps left-hand finger count to a chord choice.

This is intentionally the SIMPLEST possible version of "left hand picks the
chord": count how many fingers are extended (0-5) and map that count to one
of 5 chords. This gets a playable two-handed instrument working NOW.

Phase 3 will replace/extend this with proper per-finger chord SHAPES (e.g.
"index+middle down = G", matching specific finger combinations rather than
just a count) for a more expressive, guitar-shape-like control scheme. That
upgrade only touches this file -- main.py, audio_engine.py, sound_synth.py,
and strum_detector.py all stay exactly as they are.

GEOMETRY METHOD (count_fingers_up):
  A finger is considered "up" (extended) if its tip is farther from the
  wrist than its own base knuckle (MCP) is from the wrist. This is more
  robust to hand rotation/tilt than a fixed Y-coordinate threshold, since
  it's a relative comparison rather than an absolute screen position.
  Thumb is excluded from the count (thumb extension geometry is different
  from the other 4 fingers and less reliable with this simple method) --
  so the playable range is 0-4 fingers, mapped to 4 chords... but we want
  5 chords (Em, G, C, D, Am) to match a beginner's standard chord set, so
  "0 fingers" (closed fist) maps to a 5th chord too. See FINGER_COUNT_TO_CHORD.
"""

import math

# Landmark indices (reused from hand_tracker.py's constants -- kept duplicated
# here intentionally so this module has zero import dependency on hand_tracker,
# making it trivially portable/testable on its own).
WRIST = 0
INDEX_TIP, INDEX_MCP = 8, 5
MIDDLE_TIP, MIDDLE_MCP = 12, 9
RING_TIP, RING_MCP = 16, 13
PINKY_TIP, PINKY_MCP = 20, 17

FINGER_PAIRS = [
    (INDEX_TIP, INDEX_MCP),
    (MIDDLE_TIP, MIDDLE_MCP),
    (RING_TIP, RING_MCP),
    (PINKY_TIP, PINKY_MCP),
]

# Map finger count (0-4) to chord name. Order chosen so an open hand (4
# fingers, easiest beginner gesture) gives the most common/easiest chord.
FINGER_COUNT_TO_CHORD = {
    0: "Em",   # closed fist
    1: "G",
    2: "C",
    3: "D",
    4: "Am",
}


def _dist(p1, p2):
    return math.hypot(p1["x"] - p2["x"], p1["y"] - p2["y"])


def count_fingers_up(landmarks):
    """
    landmarks: the "landmarks" list from hand_tracker.py's output for ONE
    hand (21 dicts with x/y/z, normalized 0-1). Returns an int 0-4.
    """
    wrist = landmarks[WRIST]
    count = 0
    for tip_idx, mcp_idx in FINGER_PAIRS:
        tip = landmarks[tip_idx]
        mcp = landmarks[mcp_idx]
        if _dist(tip, wrist) > _dist(mcp, wrist):
            count += 1
    return count


def select_chord(landmarks):
    """Returns the chord name string for the given hand's landmarks."""
    count = count_fingers_up(landmarks)
    return FINGER_COUNT_TO_CHORD[count]


# ---------------------------------------------------------------------------
# Self-test with synthetic landmark data shaped like a real MediaPipe output,
# covering 0 through 4 fingers extended. Run directly: python chord_selector.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Running chord_selector.py self-test with synthetic landmark data...\n")

    def make_landmarks(fingers_up):
        """
        Builds a fake 21-point landmark list. wrist at (0.5, 0.8).
        For each of the 4 tracked fingers, places MCP at a fixed spot and
        tip either far (extended) or close (curled) based on `fingers_up`
        (a list of 4 bools for index/middle/ring/pinky).
        """
        lm = [{"x": 0.5, "y": 0.8, "z": 0.0} for _ in range(21)]  # placeholder all 21
        lm[WRIST] = {"x": 0.5, "y": 0.8, "z": 0.0}

        mcp_positions = {INDEX_MCP: (0.45, 0.6), MIDDLE_MCP: (0.5, 0.58),
                          RING_MCP: (0.55, 0.6), PINKY_MCP: (0.6, 0.62)}
        tip_pairs = [(INDEX_TIP, INDEX_MCP), (MIDDLE_TIP, MIDDLE_MCP),
                     (RING_TIP, RING_MCP), (PINKY_TIP, PINKY_MCP)]

        for mcp_idx, (mx, my) in mcp_positions.items():
            lm[mcp_idx] = {"x": mx, "y": my, "z": 0.0}

        for is_up, (tip_idx, mcp_idx) in zip(fingers_up, tip_pairs):
            mx, my = mcp_positions[mcp_idx]
            if is_up:
                # extended: tip further from wrist (smaller y = higher up the frame)
                lm[tip_idx] = {"x": mx, "y": my - 0.25, "z": 0.0}
            else:
                # curled: tip closer to wrist than the mcp is
                lm[tip_idx] = {"x": mx, "y": 0.78, "z": 0.0}

        return lm

    test_cases = [
        ([False, False, False, False], 0, "Em"),
        ([True, False, False, False], 1, "G"),
        ([True, True, False, False], 2, "C"),
        ([True, True, True, False], 3, "D"),
        ([True, True, True, True], 4, "Am"),
    ]

    all_passed = True
    for fingers_up, expected_count, expected_chord in test_cases:
        landmarks = make_landmarks(fingers_up)
        count = count_fingers_up(landmarks)
        chord = select_chord(landmarks)
        status = "OK" if (count == expected_count and chord == expected_chord) else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"  {status}  fingers_up={fingers_up} -> count={count} (expected {expected_count}), "
              f"chord={chord} (expected {expected_chord})")

    print(f"\n{'Self-test PASSED' if all_passed else 'Self-test FAILED'}: "
          f"finger counting correctly maps to chords across 0-4 fingers.")
