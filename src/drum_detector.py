"""
drum_detector.py
----------------
Air-drum hit detection. Each visible hand fires a drum sound once when ALL
of its landmarks enter a drum zone. Moving around inside the same zone stays
silent; the hand must leave that zone and enter again before it can retrigger.
"""

# Drum kit order left-to-right along the bottom of the frame.
_DRUM_ORDER = [
    "crash",
    "ride",
    "rack_tom_high",
    "rack_tom_mid",
    "hihat",
    "snare",
    "kick",
    "floor_tom",
]


def _build_bottom_row_zones():
    """Lay out large drum pads in one horizontal row at the bottom."""
    y_min, y_max = 0.66, 0.97
    x_start, x_end = 0.01, 0.99
    slot_w = (x_end - x_start) / len(_DRUM_ORDER)
    gap = 0.003

    zones = {}
    for i, name in enumerate(_DRUM_ORDER):
        x0 = x_start + i * slot_w + gap
        x1 = x_start + (i + 1) * slot_w - gap
        zones[name] = (x0, x1, y_min, y_max)
    return zones


DRUM_ZONES = _build_bottom_row_zones()


def _zone_containing_all(points):
    """Return the zone name only if every point lies fully inside it."""
    if not points:
        return None
    for name, (x_min, x_max, y_min, y_max) in DRUM_ZONES.items():
        if all(
            x_min <= x <= x_max and y_min <= y <= y_max
            for x, y in points
        ):
            return name
    return None


class _HandZoneTracker:
    """Tracks which drum zone one hand is currently fully inside."""

    def __init__(self):
        self._current_zone = None

    def update(self, points):
        zone = _zone_containing_all(points)
        entered_zone = zone is not None and zone != self._current_zone
        self._current_zone = zone
        if entered_zone:
            return zone
        return None

    def reset(self):
        self._current_zone = None


class DrumDetector:
    """Tracks both hands as independent drumsticks."""

    def __init__(self):
        self._left = _HandZoneTracker()
        self._right = _HandZoneTracker()

    def update(self, left_hand_points, right_hand_points, now=None):
        """
        left_hand_points / right_hand_points: lists of normalized (x, y)
        tuples for every landmark on that hand, or None if not visible.
        Returns a list of zone names struck this frame.
        """
        hits = []

        if left_hand_points is not None:
            zone = self._left.update(left_hand_points)
            if zone:
                hits.append(zone)
        else:
            self._left.reset()

        if right_hand_points is not None:
            zone = self._right.update(right_hand_points)
            if zone:
                hits.append(zone)
        else:
            self._right.reset()

        return hits


def _test_points_in_zone(zone_name, count=21):
    x0, x1, y0, y1 = DRUM_ZONES[zone_name]
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    return [(cx, cy) for _ in range(count)]


if __name__ == "__main__":
    detector = DrumDetector()
    events = []
    hihat_points = _test_points_in_zone("hihat")

    events.extend(detector.update(None, hihat_points))
    events.extend(detector.update(None, hihat_points))
    events.extend(detector.update(None, hihat_points))
    assert events == ["hihat"], f"expected one hihat entry hit, got {events}"

    events.extend(detector.update(None, [(0.02, 0.50)] * 21))
    events.extend(detector.update(None, hihat_points))
    assert events == ["hihat", "hihat"], f"expected retrigger after exit/re-entry, got {events}"

    partial_hihat = hihat_points[:-1] + [(0.02, 0.50)]
    events.extend(detector.update(None, partial_hihat))
    assert events == ["hihat", "hihat"], (
        f"partial hand inside zone should not trigger, got {events}"
    )

    print("Self-test PASSED: drum hits trigger once per full-hand zone entry.")
