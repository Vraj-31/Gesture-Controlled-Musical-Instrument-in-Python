"""
strum_detector.py
------------------
PHASE 2 — Strum gesture detection.

Pure motion-analysis logic: given a stream of hand positions over time,
detect "strum" events (fast directional vertical motion) and report their
direction (DOWN/UP) and intensity. Deliberately has ZERO dependency on
MediaPipe, OpenCV, or audio libraries -- it just consumes (x, y) floats.
That separation means:
  1. You can unit-test this with fake data (see the __main__ block below).
  2. Porting to JS later means re-implementing ~80 lines of plain math,
     not fighting with a different CV library's API.

ALGORITHM (velocity-threshold with debounce):
  1. Track the hand's normalized Y-coordinate (0.0 top of frame, 1.0 bottom)
     each frame.
  2. Compute vertical velocity = (y_now - y_prev) * fps  [units: frame-heights/sec]
  3. If |velocity| exceeds STRUM_VELOCITY_THRESHOLD AND we're outside the
     debounce window since the last trigger -> fire a strum event.
  4. Direction: velocity > 0 means moving DOWN the frame (Y increases
     downward in image coordinates) = "down-strum". Negative = "up-strum".

WHY VELOCITY-THRESHOLD INSTEAD OF ML/GESTURE-CLASSIFIER:
  A real guitar strum is fundamentally a fast, repeated, somewhat noisy
  motion -- it doesn't need a trained classifier to recognize, just good
  signal processing. This approach is also the cheapest computationally
  (matters because the same Python process also runs CV inference each
  frame) and the parameters are something YOU can feel and tune live by
  watching printed velocity values against your own strumming motion --
  far easier to calibrate than retraining a model.

TUNING NOTES:
  - STRUM_VELOCITY_THRESHOLD: raise it if idle hand jitter or slow
    repositioning falsely triggers strums; lower it if real strums aren't
    being detected (check printed velocity values in main.py's debug
    overlay to see what your actual strums measure).
  - DEBOUNCE_SECONDS: minimum time between two strum events. Prevents one
    physical strum motion (which spans several frames of "fast motion")
    from firing multiple notes. Real strums are usually >150ms apart even
    when played fast, so 120-150ms is a reasonable starting point.
"""

import time
from collections import deque


class StrumEvent:
    def __init__(self, direction, velocity, timestamp):
        self.direction = direction      # "DOWN" or "UP"
        self.velocity = velocity        # signed, normalized-units/sec
        self.timestamp = timestamp      # time.time() when detected

    def __repr__(self):
        return f"StrumEvent({self.direction}, v={self.velocity:.2f})"


class StrumDetector:
    def __init__(self,
                 velocity_threshold=1.8,
                 debounce_seconds=0.12,
                 smoothing_window=2):
        """
        velocity_threshold: minimum |velocity| (normalized-Y-units per second)
            to count as a strum. Normalized units means this threshold is
            resolution-independent -- same value works at 640x480 or 1920x1080.
        debounce_seconds: minimum time between two strum events.
        smoothing_window: number of recent positions averaged to reduce noise
            before computing velocity. 1 = no smoothing (raw, more responsive
            but jitter-prone). 2-3 = smoother but adds tiny latency. Keep this
            low -- too much smoothing makes strums feel "laggy" to play.
        """
        self.velocity_threshold = velocity_threshold
        self.debounce_seconds = debounce_seconds
        self.smoothing_window = max(1, smoothing_window)

        self._positions = deque(maxlen=self.smoothing_window + 1)
        self._timestamps = deque(maxlen=self.smoothing_window + 1)
        self._last_trigger_time = 0.0
        self._last_velocity = 0.0  # exposed for debug overlay / tuning

    @property
    def last_velocity(self):
        """Most recent computed velocity, useful for on-screen debug display
        so you can watch real numbers while tuning velocity_threshold."""
        return self._last_velocity

    def update(self, normalized_y, now=None):
        """
        Call this once per frame with the right hand's current normalized
        Y position (0.0 to 1.0). Returns a StrumEvent if a strum was just
        detected this frame, otherwise None.
        """
        now = now if now is not None else time.time()

        self._positions.append(normalized_y)
        self._timestamps.append(now)

        if len(self._positions) < 2:
            return None  # not enough history yet

        # Smooth by comparing the average of the most recent half vs the
        # average of the older half of our small window -- cheap and
        # avoids a single noisy frame causing a false trigger.
        positions = list(self._positions)
        timestamps = list(self._timestamps)

        dy = positions[-1] - positions[0]
        dt = timestamps[-1] - timestamps[0]
        if dt <= 0:
            return None

        velocity = dy / dt
        self._last_velocity = velocity

        if abs(velocity) < self.velocity_threshold:
            return None

        if (now - self._last_trigger_time) < self.debounce_seconds:
            return None  # still in debounce window, ignore

        self._last_trigger_time = now
        direction = "DOWN" if velocity > 0 else "UP"

        # Clear history after a trigger so the next strum is measured fresh,
        # rather than partially overlapping with the motion we just fired on.
        self._positions.clear()
        self._timestamps.clear()

        return StrumEvent(direction, velocity, now)

    def reset(self):
        self._positions.clear()
        self._timestamps.clear()
        self._last_trigger_time = 0.0


# ---------------------------------------------------------------------------
# Self-test with synthetic data. Run directly:  python strum_detector.py
# This requires no webcam, no audio, no MediaPipe -- pure logic check.
# Useful for confirming the algorithm still behaves after you tune constants.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Running strum_detector.py self-test with synthetic motion data...\n")

    detector = StrumDetector(velocity_threshold=1.5, debounce_seconds=0.1)

    fps = 30
    dt = 1.0 / fps
    t0 = 0.0

    def make_sequence():
        seq = []
        y = 0.40
        # idle
        for _ in range(10):
            seq.append(y + 0.002)
        # fast strum down
        for y2 in [0.40, 0.50, 0.60, 0.70]:
            seq.append(y2)
        # idle (holding)
        for _ in range(15):
            seq.append(0.70)
        # fast strum up
        for y2 in [0.70, 0.60, 0.50, 0.40]:
            seq.append(y2)
        # idle
        for _ in range(10):
            seq.append(0.40)
        return seq

    sequence = make_sequence()
    detected = []
    for i, y in enumerate(sequence):
        fake_now = t0 + i * dt
        event = detector.update(y, now=fake_now)
        if event:
            detected.append((i, event))

    print(f"Synthetic sequence length: {len(sequence)} frames")
    print(f"Detected {len(detected)} strum event(s):")
    for frame_idx, event in detected:
        print(f"  frame {frame_idx}: {event}")

    assert len(detected) == 2, f"Expected 2 strums, got {len(detected)}"
    assert detected[0][1].direction == "DOWN"
    assert detected[1][1].direction == "UP"
    print("\nSelf-test PASSED: detected exactly one DOWN and one UP strum.")
