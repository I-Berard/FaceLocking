# src/person_tracker.py
"""
Whole-person tracking with three-state identity model and exponential confidence decay.

This module implements:
    1. Body (whole-person) detection via OpenCV HOG (synchronous or async background)
    2. Spatial face-to-body mapping
    3. Three-state identity model (Confirmed / Temporary / Unknown)
    4. Exponential confidence decay for temporal identity continuity
    5. Target centring logic with dead-zone overlay
    6. Performance optimizations: spatial face association + periodic ArcFace verification

Three-State Model:
    CONFIRMED  — Face recognized, identity verified, confidence ≈ 1.0
    TEMPORARY  — Face lost but body still tracked, confidence decaying exponentially
    UNKNOWN    — No identity association (confidence below threshold or body lost)

Exponential Decay:
    confidence(t) = c₀ × e^(−λ × Δt)
    where:
        c₀  = confidence at last face confirmation
        Δt  = seconds since last face confirmation
        λ   = decay rate constant (higher = faster decay)

Usage:
    python -m src.person_tracker --target <name>
"""

from __future__ import annotations

import argparse
import math
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .body_features import (
    BodyFeatureExtractor,
    BodyFeatureVector,
    BodyProfile,
    compare_feature_vectors,
)
from .yolox_detector import AsyncYOLOXBodyDetector, YOLOXBodyDetector


# ─────────────────────────────────────────────────────────────
# Three-State Identity Model
# ─────────────────────────────────────────────────────────────


class IdentityState(Enum):
    """Three-state identity model for tracked persons."""

    CONFIRMED = auto()  # Face actively recognized
    TEMPORARY = auto()  # Face lost, body still tracked, confidence decaying
    UNKNOWN = auto()    # No identity association


@dataclass
class TrackedPerson:
    """Represents a tracked person with identity state and confidence."""

    track_id: int
    identity: Optional[str] = None
    state: IdentityState = IdentityState.UNKNOWN
    confidence: float = 0.0

    body_box: Optional[Tuple[int, int, int, int]] = None  # XYXY
    face_box: Optional[Tuple[int, int, int, int]] = None  # XYXY
    face_det: object = None  # FaceDet or mock

    last_confirmed_time: float = 0.0   # time.monotonic() of last face confirmation
    last_body_time: float = 0.0        # time.monotonic() of last body detection
    frames_since_face: int = 0
    body_sim: float = 0.0              # Similarity to learned body appearance
    is_reid: bool = False              # True if tracked via body re-identification (e.g. 180 turn)

    _smooth_center: Optional[np.ndarray] = field(default=None, repr=False)


# ─────────────────────────────────────────────────────────────
# Geometry Helpers
# ─────────────────────────────────────────────────────────────


def box_center(box: Tuple[int, int, int, int]) -> np.ndarray:
    """Return center (x, y) of an XYXY box."""
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)


def box_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """IoU between two XYXY boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter)


def face_inside_body(
    face_box: Tuple[int, int, int, int],
    body_box: Tuple[int, int, int, int],
    margin: float = 0.3,
) -> bool:
    """
    Check if a face center falls within a body box (with margin).

    A face is considered inside a body if its center point lies within
    the body bounding box expanded by *margin* on each side.
    """
    fc = box_center(face_box)
    bx1, by1, bx2, by2 = body_box
    bw = max(1, bx2 - bx1)
    bh = max(1, by2 - by1)

    ex1 = bx1 - margin * bw
    ey1 = by1 - margin * bh
    ex2 = bx2 + margin * bw
    ey2 = by2 + margin * bh

    return ex1 <= fc[0] <= ex2 and ey1 <= fc[1] <= ey2


# ─────────────────────────────────────────────────────────────
# Exponential Confidence Decay
# ─────────────────────────────────────────────────────────────


def exponential_decay(
    initial: float,
    elapsed: float,
    decay_rate: float,
) -> float:
    """
    Compute exponential confidence decay.

    confidence(t) = initial × e^(−λ × t)

    Args:
        initial:    Starting confidence (typically 1.0).
        elapsed:    Seconds since last confirmation.
        decay_rate: λ — higher value means faster decay.

    Returns:
        Decayed confidence clamped to [0, 1].
    """
    return float(np.clip(initial * math.exp(-decay_rate * elapsed), 0.0, 1.0))


# ─────────────────────────────────────────────────────────────
# Body Detector (OpenCV HOG)
# ─────────────────────────────────────────────────────────────


class BodyDetector:
    """Whole-person detector using OpenCV's built-in HOG + SVM."""

    def __init__(
        self,
        win_stride: Tuple[int, int] = (8, 8),
        padding: Tuple[int, int] = (8, 8),
        scale: float = 1.08,
        hit_threshold: float = 0.0,
        nms_threshold: float = 0.4,
        resize_width: int = 320,
    ):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.win_stride = win_stride
        self.padding = padding
        self.scale = scale
        self.hit_threshold = hit_threshold
        self.nms_threshold = nms_threshold
        self.resize_width = resize_width

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect persons and return XYXY bounding boxes in original frame coordinates."""
        h, w = frame.shape[:2]

        # Resize for speed
        if w > self.resize_width:
            scale_factor = self.resize_width / w
            small = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
        else:
            scale_factor = 1.0
            small = frame

        boxes, _weights = self.hog.detectMultiScale(
            small,
            winStride=self.win_stride,
            padding=self.padding,
            scale=self.scale,
            hitThreshold=self.hit_threshold,
        )

        if boxes is None or len(boxes) == 0:
            return []

        # Convert XYWH → XYXY, scale back to original resolution
        results: List[Tuple[int, int, int, int]] = []
        for x, y, bw, bh in boxes:
            x1 = int(max(0, min(x / scale_factor, w - 1)))
            y1 = int(max(0, min(y / scale_factor, h - 1)))
            x2 = int(max(0, min((x + bw) / scale_factor, w - 1)))
            y2 = int(max(0, min((y + bh) / scale_factor, h - 1)))
            results.append((x1, y1, x2, y2))

        if len(results) > 1:
            results = self._nms(results, self.nms_threshold)

        return results

    @staticmethod
    def _nms(
        boxes: List[Tuple[int, int, int, int]],
        threshold: float,
    ) -> List[Tuple[int, int, int, int]]:
        """Non-maximum suppression on XYXY boxes."""
        if not boxes:
            return []

        arr = np.array(boxes, dtype=np.float32)
        x1s, y1s, x2s, y2s = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        areas = (x2s - x1s) * (y2s - y1s)
        order = list(np.argsort(areas)[::-1])

        keep: List[int] = []
        while order:
            i = order.pop(0)
            keep.append(i)
            remaining = []
            for j in order:
                ix1 = max(x1s[i], x1s[j])
                iy1 = max(y1s[i], y1s[j])
                ix2 = min(x2s[i], x2s[j])
                iy2 = min(y2s[i], y2s[j])
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                union = areas[i] + areas[j] - inter
                if inter / max(union, 1e-6) < threshold:
                    remaining.append(j)
            order = remaining

        return [boxes[i] for i in keep]


class AsyncBodyDetector:
    """
    Asynchronous background wrapper for BodyDetector.

    Runs the CPU-heavy HOG detector in a background daemon thread
    so the main video loop never blocks or drops frames.
    """

    def __init__(self, detector: Optional[BodyDetector] = None):
        self.detector = detector or BodyDetector()
        self.is_async = True
        self._lock = threading.Lock()
        self._latest_bodies: List[Tuple[int, int, int, int]] = []
        self._pending_frame: Optional[np.ndarray] = None
        self._running = True
        self._has_new_frame = threading.Event()
        self._thread = threading.Thread(
            target=self._worker, daemon=True, name="AsyncBodyWorker"
        )
        self._thread.start()

    def _worker(self):
        while self._running:
            if not self._has_new_frame.wait(timeout=0.1):
                continue
            if not self._running:
                break
            with self._lock:
                frame = self._pending_frame
                self._pending_frame = None
                self._has_new_frame.clear()

            if frame is not None:
                try:
                    bodies = self.detector.detect(frame)
                    with self._lock:
                        self._latest_bodies = bodies
                except Exception:
                    pass

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Non-blocking detect: schedules the frame for background detection
        and immediately returns the most recent detected body boxes.
        """
        with self._lock:
            self._pending_frame = frame
            self._has_new_frame.set()
            return list(self._latest_bodies)

    def close(self):
        """Cleanly terminate the worker thread."""
        self._running = False
        self._has_new_frame.set()
        if self._thread.is_alive():
            self._thread.join(timeout=0.5)


# ─────────────────────────────────────────────────────────────
# Tracking Signal
# ─────────────────────────────────────────────────────────────


@dataclass
class TrackingSignal:
    """Position error signal for target centring."""

    error_x: float
    error_y: float
    horizontal: str   # LEFT, CENTER, RIGHT
    vertical: str     # UP, CENTER, DOWN


# ─────────────────────────────────────────────────────────────
# Person Tracker
# ─────────────────────────────────────────────────────────────


class PersonTracker:
    """
    Whole-person tracker with three-state identity model.

    Integrates:
        - Body detection (HOG / Async background)
        - Face detection & recognition
        - Spatial face-to-body mapping
        - Three-state model with exponential confidence decay
        - Target centring logic
        - Fast spatial face tracking with periodic ArcFace verification
    """

    def __init__(
        self,
        target_name: str,
        face_detector,
        embedder,
        matcher,
        body_detector=None,
        feature_extractor=None,
        reid_threshold: float = 0.75,
        decay_rate: float = 0.15,
        temporary_threshold: float = 0.25,
        verify_every: int = 15,
        ema_alpha: float = 0.30,
        dead_zone: float = 0.07,
        body_detect_interval: int = 4,
        time_fn=None,
    ):
        self.target_name = target_name
        self.face_detector = face_detector
        self.embedder = embedder
        self.matcher = matcher
        self.body_detector = body_detector or BodyDetector()
        self.feature_extractor = feature_extractor or BodyFeatureExtractor()
        self.reid_threshold = reid_threshold
        self.target_profile = BodyProfile(target_name=target_name, min_samples=8)

        self.decay_rate = decay_rate
        self.temporary_threshold = temporary_threshold
        self.verify_every = verify_every
        self.ema_alpha = ema_alpha
        self.dead_zone = dead_zone
        self.body_detect_interval = body_detect_interval
        self._time_fn = time_fn or time.monotonic

        # Tracking state
        self._next_id = 0
        self._target: Optional[TrackedPerson] = None
        self._frame_index = 0
        self._last_bodies: List[Tuple[int, int, int, int]] = []

    # ── Properties ──

    @property
    def state(self) -> IdentityState:
        """Current identity state of the tracked target."""
        if self._target is None:
            return IdentityState.UNKNOWN
        return self._target.state

    @property
    def confidence(self) -> float:
        """Current confidence of the tracked target."""
        if self._target is None:
            return 0.0
        return self._target.confidence

    @property
    def target(self) -> Optional[TrackedPerson]:
        """The currently tracked target person."""
        return self._target

    # ── Internal helpers ──

    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    @staticmethod
    def _face_box(face) -> Tuple[int, int, int, int]:
        return (face.x1, face.y1, face.x2, face.y2)

    def _identify(self, frame: np.ndarray, face):
        """Run face recognition on a detection."""
        if hasattr(face, "_cached_match") and face._cached_match is not None:
            return face._cached_match

        from .align import align_face_5pt

        aligned, _ = align_face_5pt(frame, face.kps, out_size=(112, 112))
        match = self.matcher.match(self.embedder.embed(aligned))
        face._cached_match = match
        return match

    # ── Face search & association ──

    def _find_target_face(
        self, frame: np.ndarray, faces
    ):
        """Find the target face among detections by identity (runs recognition)."""
        best = None
        best_sim = -1.0
        for face in faces:
            match = self._identify(frame, face)
            if (
                match.accepted
                and match.name == self.target_name
                and match.similarity > best_sim
            ):
                best = face
                best_sim = match.similarity
        return best

    def _associate_face(self, frame: np.ndarray, faces):
        """
        Associate a detected face with the target.

        Uses spatial continuity (IoU + displacement) when locked.
        Only runs expensive identity verification (ArcFace) when:
          - Target is recovering from TEMPORARY state (face was lost)
          - Candidate has low/no overlap with previous face (possible swap or jump)
          - It is a periodic verification frame (frame_index % verify_every == 0)
        """
        if self._target is None or not faces:
            return None

        last_face = self._target.face_box
        last_body = self._target.body_box

        if last_face is None and last_body is None:
            return None

        ranked = []
        for face in faces:
            fb = self._face_box(face)
            score = -1.0
            overlap = 0.0

            if last_face is not None:
                overlap = box_iou(last_face, fb)
                last_c = box_center(last_face)
                last_diag = max(
                    1.0,
                    math.sqrt(
                        (last_face[2] - last_face[0]) ** 2
                        + (last_face[3] - last_face[1]) ** 2
                    ),
                )
                disp = float(np.linalg.norm(box_center(fb) - last_c)) / last_diag
                score = overlap - 0.35 * disp

            # If no good match to last face, check if face is inside target body
            if score <= -0.30 and last_body is not None:
                if face_inside_body(fb, last_body):
                    score = 0.1  # Valid candidate inside body

            if score > -0.30:
                ranked.append((score, overlap, face))

        if not ranked:
            return None

        # Sort by score descending
        ranked.sort(key=lambda item: item[0], reverse=True)
        best_score, best_overlap, best_face = ranked[0]

        # Determine if identity verification (ArcFace) is required:
        # 1. Recovering from TEMPORARY / face loss
        # 2. Candidate has no overlap with previous face (possible swap or jump)
        # 3. Periodic verification frame
        needs_verify = (
            (self._target.state != IdentityState.CONFIRMED)
            or (self._target.face_box is None)
            or (best_overlap <= 0.05)
            or (self._frame_index % self.verify_every == 0)
        )

        if needs_verify:
            match = self._identify(frame, best_face)
            if not (match.accepted and match.name == self.target_name):
                # If best_face wasn't target, check other candidates in body
                for s, ov, other_face in ranked[1:]:
                    if last_body is not None and not face_inside_body(self._face_box(other_face), last_body):
                        continue
                    m = self._identify(frame, other_face)
                    if m.accepted and m.name == self.target_name:
                        return other_face
                return None

        return best_face

    def _associate_face_by_proximity(self, frame: np.ndarray, faces):
        """Backward-compatible alias for _associate_face."""
        return self._associate_face(frame, faces)

    # ── Body matching & Re-ID ──

    def _match_body_to_target_with_reid(
        self,
        bodies: List[Tuple[int, int, int, int]],
        frame: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
        """
        Find the body box matching the target using learned appearance features + spatial tracking.

        Returns (best_body_box, similarity_score).
        """
        if self._target is None or not bodies:
            return None, 0.0

        ref_box = self._target.body_box or self._target.face_box
        ref_center = box_center(ref_box) if ref_box is not None else None

        has_profile = (
            self.target_profile is not None
            and self.target_profile.sample_count > 0
            and frame is not None
            and self.feature_extractor is not None
        )

        best_body = None
        best_sim = 0.0
        best_score = -float("inf")

        for body in bodies:
            # 1. Spatial continuity score
            if ref_box is not None and ref_center is not None:
                iou_val = box_iou(ref_box, body) if self._target.body_box else 0.0
                center_dist = float(np.linalg.norm(box_center(body) - ref_center))
                bw = max(1, body[2] - body[0])
                bh = max(1, body[3] - body[1])
                diag = max(1.0, math.sqrt(bw ** 2 + bh ** 2))
                norm_dist = center_dist / diag
                spatial_score = iou_val - 0.3 * norm_dist
            else:
                spatial_score = 0.0

            # 2. Appearance Re-ID score
            app_sim = 0.0
            if has_profile:
                feat = self.feature_extractor.extract(frame, body)
                if feat is not None:
                    app_sim = self.target_profile.match(feat)

                # If appearance clearly does not match, reject candidate immediately
                # (prevents confusing an intruder/stranger for the target)
                if app_sim < self.reid_threshold * 0.80 and self.target_profile.is_ready:
                    continue

                # Combined score: appearance heavily weighted with spatial continuity prior
                score = 0.75 * app_sim + 0.25 * max(0.0, spatial_score)
            else:
                score = spatial_score
                app_sim = 1.0 if score > -0.5 else 0.0

            if score > best_score:
                best_score = score
                best_body = body
                best_sim = app_sim

        if best_body is not None:
            if has_profile and self.target_profile.is_ready:
                if best_sim >= self.reid_threshold:
                    return best_body, best_sim
                return None, best_sim
            elif best_score > -0.5:
                return best_body, best_sim

        return None, 0.0

    def _match_body_to_target(
        self,
        bodies: List[Tuple[int, int, int, int]],
        frame: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[int, int, int, int]]:
        """Find the body box that best matches the target's last known position."""
        matched, _ = self._match_body_to_target_with_reid(bodies, frame=frame)
        return matched

    # ── Face-to-body mapping ──

    def _map_faces_to_bodies(
        self,
        faces,
        bodies: List[Tuple[int, int, int, int]],
    ) -> Dict[int, list]:
        """Map face detections to body bounding boxes by spatial containment."""
        mapping: Dict[int, list] = {i: [] for i in range(len(bodies))}

        for face in faces:
            fb = self._face_box(face)
            for i, body in enumerate(bodies):
                if face_inside_body(fb, body):
                    mapping[i].append(face)
                    break

        return mapping

    # ── Main update ──

    def update(
        self, frame: np.ndarray
    ) -> Tuple[Optional[TrackedPerson], Optional[TrackingSignal]]:
        """
        Process a frame and update tracking state.

        State transitions:
            UNKNOWN   → CONFIRMED   (target face found)
            CONFIRMED → TEMPORARY   (face lost, body present, confidence above threshold)
            CONFIRMED → UNKNOWN     (face & body lost, or confidence too low)
            TEMPORARY → CONFIRMED   (face re-acquired)
            TEMPORARY → UNKNOWN     (confidence decayed below threshold)

        Returns:
            (tracked_person, tracking_signal) — or (None, None) if target not found.
        """
        self._frame_index += 1
        now = self._time_fn()

        # ── Detect faces (fast Haar cascade) ──
        faces = self.face_detector.detect(frame, max_faces=8)

        # ── Detect bodies (periodically or async background) ──
        if self._frame_index % self.body_detect_interval == 0:
            self._last_bodies = self.body_detector.detect(frame)
        bodies = self._last_bodies

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # STATE MACHINE
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        if self._target is None or self._target.state == IdentityState.UNKNOWN:
            # ── SEARCHING: find and confirm target ──
            target_face = self._find_target_face(frame, faces)

            if target_face is not None:
                fb = self._face_box(target_face)

                matched_body = None
                for body in bodies:
                    if face_inside_body(fb, body):
                        matched_body = body
                        break

                self._target = TrackedPerson(
                    track_id=self._new_id(),
                    identity=self.target_name,
                    state=IdentityState.CONFIRMED,
                    confidence=1.0,
                    body_box=matched_body,
                    face_box=fb,
                    face_det=target_face,
                    last_confirmed_time=now,
                    last_body_time=now,
                    is_reid=False,
                    body_sim=1.0,
                )
                if matched_body is not None and frame is not None and self.feature_extractor is not None:
                    feat = self.feature_extractor.extract(frame, matched_body)
                    if feat is not None:
                        self.target_profile.update(feat)

                return self._target, self._position_signal(frame.shape, fb)

            return None, None

        # ── Target exists — maintain tracking ──

        # Step 1: try to associate target face (fast spatial association)
        target_face = self._associate_face(frame, faces)
        if target_face is None:
            target_face = self._find_target_face(frame, faces)

        if target_face is not None:
            # ── CONFIRMED: face found and recognized ──
            fb = self._face_box(target_face)

            matched_body, _ = self._match_body_to_target_with_reid(bodies, frame=frame)
            if matched_body is None:
                for body in bodies:
                    if face_inside_body(fb, body):
                        matched_body = body
                        break

            prev_fb = self._target.face_box
            self._target.state = IdentityState.CONFIRMED
            self._target.confidence = 1.0
            self._target.face_box = fb
            self._target.face_det = target_face
            self._target.last_confirmed_time = now
            self._target.last_body_time = now
            self._target.frames_since_face = 0
            self._target.is_reid = False
            self._target.body_sim = 1.0

            if matched_body is not None:
                self._target.body_box = matched_body
                # Actively learn body features while face is confirmed inside body
                if frame is not None and self.feature_extractor is not None:
                    feat = self.feature_extractor.extract(frame, matched_body)
                    if feat is not None:
                        self.target_profile.update(feat)
            elif self._target.body_box is not None and prev_fb is not None:
                # Shift previous body box with face motion for smooth display
                dx = fb[0] - prev_fb[0]
                dy = fb[1] - prev_fb[1]
                bx1, by1, bx2, by2 = self._target.body_box
                self._target.body_box = (bx1 + dx, by1 + dy, bx2 + dx, by2 + dy)

            return self._target, self._position_signal(frame.shape, fb)

        # Step 2: face not found — check if body is still tracked or re-identified (e.g. 180° rotation)
        self._target.frames_since_face += 1
        self._target.face_det = None
        self._target.face_box = None

        matched_body, body_sim = self._match_body_to_target_with_reid(bodies, frame=frame)

        if matched_body is not None:
            self._target.body_box = matched_body
            self._target.last_body_time = now
            self._target.body_sim = body_sim

            if self.target_profile.is_ready and body_sim >= self.reid_threshold:
                # ── 180° RE-ID: body recognized by appearance features ──
                self._target.is_reid = True
                self._target.confidence = min(0.92, max(0.65, float(body_sim * 0.90)))
                self._target.state = IdentityState.TEMPORARY
                self._target.last_confirmed_time = now
                return self._target, self._position_signal(frame.shape, matched_body, is_body=True)
            else:
                self._target.is_reid = False
                # If target profile is ready and candidate does not match, reject stranger!
                if self.target_profile.is_ready:
                    self._target.state = IdentityState.UNKNOWN
                    self._target.identity = None
                    self._target.confidence = 0.0
                    return None, None

                # Profile not yet ready: fallback to exponential decay
                elapsed = now - self._target.last_confirmed_time
                self._target.confidence = exponential_decay(
                    1.0, elapsed, self.decay_rate
                )

                if self._target.confidence >= self.temporary_threshold:
                    self._target.state = IdentityState.TEMPORARY
                    return self._target, self._position_signal(frame.shape, matched_body, is_body=True)
                else:
                    self._target.state = IdentityState.UNKNOWN
                    self._target.identity = None
                    self._target.confidence = 0.0
                    return None, None

        # Step 3: both face and body lost
        elapsed = now - self._target.last_confirmed_time
        self._target.confidence = exponential_decay(
            1.0, elapsed, self.decay_rate
        )
        self._target.body_box = None

        if self._target.confidence >= self.temporary_threshold:
            self._target.state = IdentityState.TEMPORARY
            return self._target, None
        else:
            self._target.state = IdentityState.UNKNOWN
            self._target.identity = None
            self._target.confidence = 0.0
            return None, None

    # ── Centring signal ──

    def _position_signal(
        self,
        shape: Tuple[int, ...],
        box: Tuple[int, int, int, int],
        is_body: bool = False,
    ) -> TrackingSignal:
        """Compute centring error signal relative to frame centre."""
        height, width = shape[:2]
        if is_body:
            # Target upper 20% of body box (head/neck area) for smooth head-turn tracking
            bx1, by1, bx2, by2 = box
            c = np.array([(bx1 + bx2) / 2.0, by1 + (by2 - by1) * 0.20], dtype=np.float32)
        else:
            c = box_center(box)

        if self._target is not None:
            if self._target._smooth_center is None:
                self._target._smooth_center = c
            else:
                a = self.ema_alpha
                self._target._smooth_center = a * c + (1.0 - a) * self._target._smooth_center
            c = self._target._smooth_center

        ex = float((c[0] - width / 2.0) / (width / 2.0))
        ey = float((c[1] - height / 2.0) / (height / 2.0))

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


# ─────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────

STATE_COLORS = {
    IdentityState.CONFIRMED: (0, 220, 0),     # Green
    IdentityState.TEMPORARY: (0, 180, 255),   # Orange
    IdentityState.UNKNOWN:   (0, 0, 220),     # Red
}


def draw_label(frame, text, xy, color, scale=0.62):
    """Draw text with a dark shadow for visibility."""
    cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────
# Main demo
# ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Whole-person tracking with three-state identity model"
    )
    parser.add_argument("--target", required=True, help="enrolled identity to track")
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.81)
    parser.add_argument("--decay-rate", type=float, default=0.15, help="exponential decay λ")
    parser.add_argument("--verify-every", type=int, default=15, help="re-verify face identity every N frames")
    parser.add_argument("--body-interval", type=int, default=4, help="body detection interval in frames")
    parser.add_argument("--detector", choices=["yolox", "hog"], default="yolox", help="body detector backend")
    parser.add_argument("--yolox-model", type=str, default="models/yolox_nano.onnx", help="path to YOLOX ONNX model")
    parser.add_argument("--reid-threshold", type=float, default=0.75, help="body feature Re-ID similarity threshold")
    parser.add_argument("--sync-body", action="store_true", help="run body detection synchronously (no background thread)")
    parser.add_argument("--mqtt", action="store_true", help="enable MQTT publishing")
    parser.add_argument("--mqtt-host", type=str, default="localhost")
    parser.add_argument("--mqtt-port", type=int, default=1883)
    args = parser.parse_args()

    from .embed import ArcFaceEmbedderONNX
    from .recognize import FaceDBMatcher, HaarFaceLandmarker5pt, load_db_npz

    # Face pipeline
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

    # Body detector (YOLOX by default, or HOG fallback)
    if args.detector == "yolox":
        if args.sync_body:
            body_detector = YOLOXBodyDetector(model_path=args.yolox_model)
        else:
            body_detector = AsyncYOLOXBodyDetector(model_path=args.yolox_model)
    else:
        if args.sync_body:
            body_detector = BodyDetector()
        else:
            body_detector = AsyncBodyDetector()

    # Person tracker with online feature learning & Re-ID
    tracker = PersonTracker(
        target_name=args.target,
        face_detector=detector,
        embedder=embedder,
        matcher=matcher,
        body_detector=body_detector,
        reid_threshold=args.reid_threshold,
        decay_rate=args.decay_rate,
        verify_every=args.verify_every,
        body_detect_interval=args.body_interval,
    )

    # Face signals (optional)
    from .face_signals import FaceSignalExtractor

    signals = FaceSignalExtractor()

    # MQTT (optional)
    mqtt_client = None
    if args.mqtt:
        try:
            import paho.mqtt.client as mqtt

            mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
            mqtt_client.connect(args.mqtt_host, args.mqtt_port, 60)
            print(f"[person_tracker] MQTT connected to {args.mqtt_host}:{args.mqtt_port}")
        except Exception as e:
            print(f"[person_tracker] MQTT connection failed: {e}")
            mqtt_client = None

    # Camera
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

    # FPS tracking
    fps_start = time.time()
    fps_frames = 0
    fps_display = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            fps_frames += 1
            now_t = time.time()
            if now_t - fps_start >= 0.5:
                fps_display = fps_frames / (now_t - fps_start)
                fps_frames = 0
                fps_start = now_t

            person, position = tracker.update(frame)
            view = frame.copy()

            # ── State header ──
            state = tracker.state
            conf = tracker.confidence
            state_color = STATE_COLORS.get(state, (128, 128, 128))
            state_text = f"{state.name}: {args.target} (conf={conf:.0%})"
            draw_label(view, state_text, (12, 28), state_color, 0.72)

            # ── FPS counter ──
            draw_label(view, f"{fps_display:.1f} FPS", (view.shape[1] - 130, 28), (200, 200, 200), 0.55)

            # ── Draw whole body box (including face) ──
            if person is not None and person.body_box is not None:
                bx1, by1, bx2, by2 = person.body_box
                if person.state == IdentityState.CONFIRMED:
                    box_color = (0, 220, 0)  # Bright Green
                    box_label = f"TARGET: {args.target} ({conf:.0%})"
                elif getattr(person, "is_reid", False):
                    box_color = (255, 200, 0)  # Cyan
                    box_label = f"180 RE-ID: {args.target} (sim={person.body_sim:.0%})"
                else:
                    box_color = (0, 180, 255)  # Orange
                    box_label = f"BODY ({conf:.0%})"

                cv2.rectangle(view, (bx1, by1), (bx2, by2), box_color, 3)
                draw_label(view, box_label, (bx1, max(22, by1 - 8)), box_color, 0.60)

            # ── Draw face box inside whole body + signals ──
            if person is not None and person.face_det is not None:
                face = person.face_det
                x1, y1, x2, y2 = face.x1, face.y1, face.x2, face.y2
                cv2.rectangle(view, (x1, y1), (x2, y2), state_color, 2)

                face_state = signals.analyze(frame, (x1, y1, x2, y2))
                if face_state is not None:
                    if face_state.blink:
                        blink_total += 1
                    expression = "SMILE" if face_state.smiling else "NEUTRAL"
                    eye_text = "EYES CLOSED" if face_state.eyes_closed else "EYES OPEN"
                    draw_label(view, expression, (x1, max(55, y1 - 40)), (0, 255, 255), 0.5)
                    draw_label(
                        view,
                        f"{eye_text} blinks={blink_total}",
                        (x1, max(75, y1 - 20)),
                        (255, 255, 0),
                        0.5,
                    )

            elif person is not None and person.state == IdentityState.TEMPORARY:
                # Face hidden but body tracked
                if person.body_box is not None:
                    bx1, by1, bx2, by2 = person.body_box
                    cx = (bx1 + bx2) // 2
                    cy = by1 + 30
                    if getattr(person, "is_reid", False):
                        status_msg = f"FACE HIDDEN - 180 RE-ID (sim={person.body_sim:.0%})"
                        tag_color = (255, 200, 0)
                    else:
                        status_msg = f"FACE HIDDEN ({conf:.0%})"
                        tag_color = (0, 180, 255)
                    draw_label(
                        view,
                        status_msg,
                        (cx - 100, cy),
                        tag_color,
                        0.55,
                    )
            else:
                signals.reset()

            # ── Feature Learning status overlay ──
            if tracker.target_profile.is_ready:
                learn_text = f"BODY PROFILE LOCKED ({len(tracker.target_profile.samples)} samples)"
                draw_label(view, learn_text, (12, view.shape[0] - 65), (0, 220, 0), 0.52)
            else:
                learn_text = (
                    f"LEARNING BODY: {tracker.target_profile.learning_progress:.0%} "
                    f"({tracker.target_profile.sample_count}/{tracker.target_profile.min_samples})"
                )
                draw_label(view, learn_text, (12, view.shape[0] - 65), (0, 200, 255), 0.52)

            # ── Position signal ──
            if position is not None:
                draw_label(
                    view,
                    f"H={position.horizontal} V={position.vertical} "
                    f"error=({position.error_x:+.2f},{position.error_y:+.2f})",
                    (12, 56),
                    (255, 170, 0),
                    0.60,
                )
                if mqtt_client is not None:
                    mqtt_client.publish("position", str(position.error_y))

            # ── Dead zone overlay ──
            h, w = view.shape[:2]
            dz = tracker.dead_zone
            cv2.rectangle(
                view,
                (int(w * (0.5 - dz / 2)), int(h * (0.5 - dz / 2))),
                (int(w * (0.5 + dz / 2)), int(h * (0.5 + dz / 2))),
                (120, 120, 120),
                1,
            )

            # ── Confidence bar ──
            bar_x, bar_y, bar_w, bar_h = 12, h - 40, 200, 16
            cv2.rectangle(view, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
            fill_w = int(bar_w * conf)
            cv2.rectangle(view, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), state_color, -1)
            cv2.rectangle(
                view, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (180, 180, 180), 1
            )
            draw_label(view, "Confidence", (bar_x, bar_y - 4), (200, 200, 200), 0.45)

            cv2.imshow("Person Tracking (3-State)", view)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        cap.release()
        signals.close()
        cv2.destroyAllWindows()
        if hasattr(body_detector, "close"):
            body_detector.close()
        if mqtt_client is not None:
            mqtt_client.disconnect()


if __name__ == "__main__":
    main()
