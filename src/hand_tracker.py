"""
hand_tracker.py
----------------
Core hand-tracking wrapper around MediaPipe's HandLandmarker (Tasks API).

IMPORTANT: MediaPipe removed the old `mp.solutions.hands` API in recent
versions (0.10.x). This code uses the current `mediapipe.tasks` API instead.
This is actually good for your eventual web port: the JS equivalent
(`@mediapipe/tasks-vision`'s HandLandmarker) is the SAME API family, with
near-identical method names and a near-identical model file. So the logic
you build on top of this (chord detection, strum detection) will translate
to JS more directly than it would have with the old API.

This module ONLY does detection + simple packaging of results. It knows
nothing about chords, strums, or audio -- that separation keeps Phase 2/3
logic clean and testable on its own.

SETUP REQUIRED (one-time): this API needs a model file downloaded to disk.
Run this once before using this module:

    curl -o hand_landmarker.task -L https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

(or it will be auto-downloaded by setup.py / the instructions in README.md)
"""

import os
import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

# Landmark indices we care about most (21 total per hand).
# Full reference: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
INDEX_MCP = 5
MIDDLE_TIP = 12
MIDDLE_MCP = 9
RING_TIP = 16
RING_MCP = 13
PINKY_TIP = 20
PINKY_MCP = 17

# Hand connections for drawing the skeleton (pairs of landmark indices).
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (0, 9), (9, 10), (10, 11), (11, 12),     # middle
    (0, 13), (13, 14), (14, 15), (15, 16),   # ring
    (0, 17), (17, 18), (18, 19), (19, 20),   # pinky
    (5, 9), (9, 13), (13, 17),               # palm
]


class HandTracker:
    def __init__(self, max_hands=2, detection_confidence=0.6, tracking_confidence=0.5,
                 model_path=MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"\n\nModel file not found at: {model_path}\n"
                "Download it once with:\n\n"
                "  curl -o hand_landmarker.task -L "
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                "hand_landmarker/float16/1/hand_landmarker.task\n\n"
                "Run that command from inside the src/ folder, then try again."
            )

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,  # VIDEO mode = expects increasing timestamps, good for webcam streams
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.landmarker = HandLandmarker.create_from_options(options)
        self._frame_timestamp_ms = 0

    def process(self, frame_bgr):
        """
        Takes a BGR OpenCV frame, returns a list of hand dicts:
        [
          {
            "label": "Left" or "Right",  # corrected for mirrored/selfie view, see note below
            "score": 0.97,
            "landmarks": [ {"x":.., "y":.., "z":..}, ... 21 points, normalized 0-1 ],
            "landmarks_px": [ (x_px, y_px), ... 21 points, pixel coords ]
          },
          ...
        ]

        NOTE ON LEFT/RIGHT LABELING:
        MediaPipe's handedness label assumes a non-mirrored camera (i.e. it
        labels hands as if you're looking AT the person, not as if you're
        looking in a mirror). Since main.py flips the frame for a natural
        "mirror" preview before calling process(), the label MediaPipe
        returns ends up matching the user's actual left/right hand
        correctly. If you ever remove the flip in main.py, swap the labels
        here instead of hunting through the rest of the codebase.
        """
        h, w, _ = frame_bgr.shape
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        self._frame_timestamp_ms += 33  # assume ~30fps; fine since VIDEO mode only needs monotonic increase
        result = self.landmarker.detect_for_video(mp_image, self._frame_timestamp_ms)

        hands_out = []
        if result.hand_landmarks:
            for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                raw_label = handedness[0].category_name  # "Left" or "Right"
                # This webcam setup reports mirrored hands opposite to how the
                # player sees them, so swap labels once at the source.
                label = "Right" if raw_label == "Left" else "Left"
                score = handedness[0].score

                landmarks = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand_landmarks]
                landmarks_px = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

                hands_out.append({
                    "label": label,
                    "score": score,
                    "landmarks": landmarks,
                    "landmarks_px": landmarks_px,
                })
        return hands_out

    def draw(self, frame_bgr, hands_data):
        """Draws landmarks + connections + label for debug/visual feedback."""
        for hand in hands_data:
            pts = hand["landmarks_px"]

            # draw connections
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame_bgr, pts[a], pts[b], (0, 200, 0), 2)

            # draw points
            for (x, y) in pts:
                cv2.circle(frame_bgr, (x, y), 4, (0, 100, 255), -1)

            # label near wrist
            wrist_px = pts[WRIST]
            color = (0, 255, 0) if hand["label"] == "Right" else (255, 150, 0)
            cv2.putText(
                frame_bgr,
                f'{hand["label"]} ({hand["score"]:.2f})',
                (wrist_px[0] - 20, wrist_px[1] + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )
        return frame_bgr

    def close(self):
        self.landmarker.close()
