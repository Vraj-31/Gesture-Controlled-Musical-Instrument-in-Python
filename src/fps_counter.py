"""
fps_counter.py
--------------
Tiny utility to measure and display FPS. Useful because hand tracking
performance varies a lot by machine -- you want to know early if you're
getting 30fps (great) or 8fps (you'll need to lower camera resolution
or max_hands, or it will hurt strum-detection timing in Phase 2).
"""

import time


class FPSCounter:
    def __init__(self, smoothing=0.9):
        self._prev_time = time.time()
        self._fps = 0.0
        self._smoothing = smoothing

    def update(self):
        now = time.time()
        dt = now - self._prev_time
        self._prev_time = now
        if dt > 0:
            current_fps = 1.0 / dt
            self._fps = (self._fps * self._smoothing) + (current_fps * (1 - self._smoothing))
        return self._fps

    @property
    def fps(self):
        return self._fps
