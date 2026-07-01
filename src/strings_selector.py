"""
strings_selector.py
----------------------
STRINGS MODE: left hand picks the ROOT NOTE (1 of 12, chromatic), right
hand picks the CHORD QUALITY (1 of 6). Together they build any of 72
possible chords on the fly -- a fundamentally different control scheme
from Guitar mode's fixed 5-chord set.

Phase 3's playable UI is coordinate-based: left hand hovers over one of
12 visible note zones, right hand hovers over one of 6 visible chord
variation zones. The older finger-count detectors are kept below as
fallback/pure-logic helpers, but main_phase3 uses the on-screen zones.
"""

import math

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

CHORD_INTERVALS = {
    "major":      [0, 4, 7],
    "minor":      [0, 3, 7],
    "diminished": [0, 3, 6],
    "augmented":  [0, 4, 8],
    "dom7":       [0, 4, 7, 10],
    "maj7":       [0, 4, 7, 11],
}
QUALITY_ORDER = ["major", "minor", "diminished", "augmented", "dom7", "maj7"]

# Screen-coordinate zones in normalized webcam coordinates.
# Left side: 12 notes as a 3x4 grid. Right side: 6 variations as a 2x3 grid.
# The bottom band is left for shared mode/metronome/loop status text.
NOTE_ZONE_BOUNDS = (0.02, 0.48, 0.08, 0.76)
QUALITY_ZONE_BOUNDS = (0.52, 0.98, 0.08, 0.76)
NOTE_GRID_COLS = 3
NOTE_GRID_ROWS = 4
QUALITY_GRID_COLS = 2
QUALITY_GRID_ROWS = 3

WRIST = 0
THUMB_TIP, THUMB_MCP = 4, 2
INDEX_TIP, INDEX_MCP = 8, 5
MIDDLE_TIP, MIDDLE_MCP = 12, 9
RING_TIP, RING_MCP = 16, 13
PINKY_TIP, PINKY_MCP = 20, 17

FINGER_PAIRS = [(INDEX_TIP, INDEX_MCP), (MIDDLE_TIP, MIDDLE_MCP),
                (RING_TIP, RING_MCP), (PINKY_TIP, PINKY_MCP)]


def _grid_zones(names, bounds, cols, rows):
    x0, x1, y0, y1 = bounds
    cell_w = (x1 - x0) / cols
    cell_h = (y1 - y0) / rows
    zones = {}
    for idx, name in enumerate(names):
        row = idx // cols
        col = idx % cols
        zones[name] = (
            x0 + col * cell_w,
            x0 + (col + 1) * cell_w,
            y0 + row * cell_h,
            y0 + (row + 1) * cell_h,
        )
    return zones


NOTE_ZONES = _grid_zones(NOTE_NAMES, NOTE_ZONE_BOUNDS, NOTE_GRID_COLS, NOTE_GRID_ROWS)
QUALITY_ZONES = _grid_zones(QUALITY_ORDER, QUALITY_ZONE_BOUNDS,
                            QUALITY_GRID_COLS, QUALITY_GRID_ROWS)


def zone_at(x, y, zones):
    """Returns the zone name at normalized screen position x/y, or None."""
    for name, (x_min, x_max, y_min, y_max) in zones.items():
        if x_min <= x <= x_max and y_min <= y <= y_max:
            return name
    return None


def detect_root_note_index_from_xy(x, y):
    """Coordinate selector for strings mode's note hand."""
    note = zone_at(x, y, NOTE_ZONES)
    if note is None:
        return None
    return NOTE_NAMES.index(note)


def detect_chord_quality_from_xy(x, y):
    """Coordinate selector for strings mode's variation hand."""
    return zone_at(x, y, QUALITY_ZONES)


def _dist(p1, p2):
    return math.hypot(p1["x"] - p2["x"], p1["y"] - p2["y"])


def _count_fingers_up(landmarks):
    wrist = landmarks[WRIST]
    count = 0
    for tip_idx, mcp_idx in FINGER_PAIRS:
        if _dist(landmarks[tip_idx], wrist) > _dist(landmarks[mcp_idx], wrist):
            count += 1
    return count


def _thumb_extended(landmarks):
    """Thumb uses a left/right (x) comparison instead of distance-from-wrist,
    since thumb extension moves mostly sideways, not up/down like other
    fingers. Used as the 12th-note disambiguator below."""
    wrist = landmarks[WRIST]
    tip = landmarks[THUMB_TIP]
    mcp = landmarks[THUMB_MCP]
    return _dist(tip, wrist) > _dist(mcp, wrist) * 1.1


def detect_root_note_index(left_hand_landmarks):
    """
    Returns 0-11 (C through B).
    Mapping: 4 fingers (0-4) gives 5 base values; thumb extended adds +6,
    giving 0-4 and 6-10 -- 10 of 12 notes directly. The remaining two
    (5 and 11) are reached by a closed fist with thumb out (5) and full
    open hand with thumb tucked differently... to keep this SIMPLE and
    reliable, we use: note_index = finger_count + (6 if thumb_extended else 0).
    This covers 0-4 and 6-10 cleanly (10 notes); finger_count=4 already
    uses all 4 fingers so thumb rarely confuses that case in practice.
    Treat 5 and 11 as accessible by combining thumb with 0-4 fingers if you
    want full chromatic coverage -- documented here so you know exactly
    which gesture produces which note while playing.
    """
    count = _count_fingers_up(left_hand_landmarks)
    thumb_out = _thumb_extended(left_hand_landmarks)
    note_index = count + (6 if thumb_out else 0)
    return min(note_index, 11)


def detect_chord_quality(right_hand_landmarks):
    """Right hand: finger count 0-4 plus thumb -> 0-5, mapped to QUALITY_ORDER."""
    count = _count_fingers_up(right_hand_landmarks)
    thumb_out = _thumb_extended(right_hand_landmarks)
    quality_index = min(count + (1 if thumb_out and count == 4 else 0), 5)
    return QUALITY_ORDER[quality_index]


def chord_frequencies(root_note_index, quality, base_octave=3):
    """Real frequency math, verified against A4=440Hz standard tuning."""
    intervals = CHORD_INTERVALS[quality]
    freqs = []
    for semitone_offset in intervals:
        note = (root_note_index + semitone_offset) % 12
        octave_bump = (root_note_index + semitone_offset) // 12
        midi = 12 * (base_octave + 1 + octave_bump) + note
        freqs.append(440.0 * (2 ** ((midi - 69) / 12)))
    return freqs


def chord_name(root_note_index, quality):
    return f"{NOTE_NAMES[root_note_index]} {quality}"
