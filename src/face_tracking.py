# src/face_tracking.py
"""
Target-locked face tracking pipeline with real-time signal analysis (EAR, blinks, expressions).

Usage:
    python -m src.face_tracking --target <name>
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from .align import align_face_5pt
from .embed import ArcFaceEmbedderONNX
from .face_signals import FaceSignalExtractor
from .haar_5pt import Haar5ptDetector
from .recognize import FaceDBMatcher, HaarFaceLandmarker5pt, load_db_npz


class LockState(Enum):
    SEARCHING = auto()
    LOCKED = auto()
    LOST = auto()


def iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """Calculate Intersection over Union (IoU) between two XYXY boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter)


def center(box: Tuple[int, int, int, int]) -> np.ndarray:
    """Return center (x, y) coordinates of an XYXY box."""
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)


@dataclass
class TrackingSignal:
    error_x: float
    error_y: float
    horizontal: str
    vertical: str


class LockedFaceTracker:
    def __init__(
        self,
        target_name: str,
        detector,
        embedder,
        matcher,
        verify_every: int = 10,
        lost_timeout: int = 24,
        ema_alpha: float = 0.30,
        dead_zone: float = 0.07,
    ):
        self.target_name = target_name
        self.detector = detector
        self.embedder = embedder
        self.matcher = matcher
        self.verify_every = verify_every
        self.lost_timeout = lost_timeout
        self.ema_alpha = ema_alpha
        self.dead_zone = dead_zone

        self.state = LockState.SEARCHING
        self.last_box: Optional[Tuple[int, int, int, int]] = None
        self.smooth_center: Optional[np.ndarray] = None
        self.lost_frames = 0
        self.frame_index = 0

    @staticmethod
    def box(face) -> Tuple[int, int, int, int]:
        return (face.x1, face.y1, face.x2, face.y2)

    def identity(self, frame, face):
        aligned, _ = align_face_5pt(frame, face.kps, out_size=(112, 112))
        return self.matcher.match(self.embedder.embed(aligned))

    def target_is_verified(self, frame, face) -> bool:
        match = self.identity(frame, face)
        return bool(match.accepted and match.name == self.target_name)

    def acquire(self, frame, faces):
        best = None
        best_similarity = -1.0
        for face in faces:
            match = self.identity(frame, face)
            if (
                match.accepted
                and match.name == self.target_name
                and match.similarity > best_similarity
            ):
                best, best_similarity = face, match.similarity
        return best

    def associate(self, faces):
        if self.last_box is None or not faces:
            return None
        last_center = center(self.last_box)
        last_diag = max(
            np.linalg.norm(
                np.array(
                    [
                        self.last_box[2] - self.last_box[0],
                        self.last_box[3] - self.last_box[1],
                    ],
                    dtype=np.float32,
                )
            ),
            1.0,
        )
        ranked = []
        for face in faces:
            box = self.box(face)
            overlap = iou(self.last_box, box)
            displacement = np.linalg.norm(center(box) - last_center) / last_diag
            score = overlap - 0.35 * displacement
            ranked.append((score, face))

        if not ranked:
            return None

        score, candidate = max(ranked, key=lambda item: item[0])
        return candidate if score > -0.30 else None

    def update(self, frame):
        self.frame_index += 1
        faces = self.detector.detect(frame, max_faces=8)

        if self.state == LockState.SEARCHING:
            candidate = self.acquire(frame, faces)
        else:
            candidate = self.associate(faces)

        if (
            candidate is not None
            and (
                self.state == LockState.LOST
                or self.frame_index % self.verify_every == 0
            )
            and not self.target_is_verified(frame, candidate)
        ):
            candidate = None

        if candidate is None:
            self.lost_frames += 1
            if self.last_box is not None:
                self.state = LockState.LOST
            if self.lost_frames > self.lost_timeout:
                self.state = LockState.SEARCHING
                self.last_box = None
                self.smooth_center = None
            return None, None

        self.state = LockState.LOCKED
        self.lost_frames = 0
        self.last_box = self.box(candidate)
        raw_center = center(self.last_box)

        if self.smooth_center is None:
            self.smooth_center = raw_center
        else:
            a = self.ema_alpha
            self.smooth_center = a * raw_center + (1.0 - a) * self.smooth_center

        return candidate, self.position_signal(frame.shape)

    def position_signal(self, shape) -> TrackingSignal:
        height, width = shape[:2]
        ex = float((self.smooth_center[0] - width / 2.0) / (width / 2.0))
        ey = float((self.smooth_center[1] - height / 2.0) / (height / 2.0))
        horizontal = "CENTER"
        vertical = "CENTER"

        if ex < -self.dead_zone:
            horizontal = "LEFT"
        elif ex > self.dead_zone:
            horizontal = "RIGHT"

        if ey < -self.dead_zone:
            vertical = "UP"
        elif ey > self.dead_zone:
            vertical = "DOWN"

        return TrackingSignal(ex, ey, horizontal, vertical)


def draw_label(frame, text, xy, color, scale=0.62):
    cv2.putText(
        frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA
    )
    cv2.putText(
        frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, help="enrolled identity to lock")
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.81)
    args = parser.parse_args()

    detector = HaarFaceLandmarker5pt(min_size=(70, 70), debug=False)
    embedder = ArcFaceEmbedderONNX(
        model_path="models/embedder_arcface.onnx",
        input_size=(112, 112),
        debug=False,
    )
    matcher = FaceDBMatcher(
        load_db_npz(Path("data/db/face_db.npz")),
        dist_thresh=args.threshold,
    )

    tracker = LockedFaceTracker(args.target, detector, embedder, matcher)
    signals = FaceSignalExtractor()

    cap = None
    for cam_idx in [args.camera, 1, 0, 2]:
        temp_cap = cv2.VideoCapture(cam_idx)
        if temp_cap.isOpened():
            cap = temp_cap
            break
        temp_cap.release()

    if cap is None or not cap.isOpened():
        raise RuntimeError(f"Camera not available (checked {args.camera}, 1, 0, 2)")

    blink_total = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            locked_face, position = tracker.update(frame)
            view = frame.copy()

            state_text = f"{tracker.state.name}: {args.target}"
            state_color = (0, 180, 0) if locked_face is not None else (0, 140, 255)
            draw_label(view, state_text, (12, 28), state_color, 0.72)

            if locked_face is not None:
                box = tracker.box(locked_face)
                x1, y1, x2, y2 = box
                cv2.rectangle(view, (x1, y1), (x2, y2), (255, 170, 0), 3)

                face_state = signals.analyze(frame, box)
                if face_state is not None:
                    if face_state.blink:
                        blink_total += 1

                    expression = "SMILE" if face_state.smiling else "NEUTRAL"
                    eye_text = "EYES CLOSED" if face_state.eyes_closed else "EYES OPEN"

                    draw_label(view, expression, (x1, max(55, y1 - 50)), (0, 255, 255))
                    draw_label(
                        view,
                        f"{eye_text} blinks={blink_total}",
                        (x1, max(78, y1 - 25)),
                        (255, 255, 0),
                    )
                    draw_label(
                        view,
                        f"EAR={face_state.ear:.3f} smile_conf={face_state.smile_score * 100:.0f}%",
                        (12, view.shape[0] - 18),
                        (255, 255, 255),
                        0.52,
                    )

                if position is not None:
                    draw_label(
                        view,
                        f"H={position.horizontal} V={position.vertical} "
                        f"error=({position.error_x:+.2f},{position.error_y:+.2f})",
                        (12, 56),
                        (255, 170, 0),
                        0.60,
                    )
            else:
                signals.reset()

            h, w = view.shape[:2]
            dz = tracker.dead_zone
            cv2.rectangle(
                view,
                (int(w * (0.5 - dz / 2)), int(h * (0.5 - dz / 2))),
                (int(w * (0.5 + dz / 2)), int(h * (0.5 + dz / 2))),
                (120, 120, 120),
                1,
            )

            cv2.imshow("Locked Face Tracking", view)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        cap.release()
        signals.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()