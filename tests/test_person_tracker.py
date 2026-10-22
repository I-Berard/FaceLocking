# tests/test_person_tracker.py
"""
Comprehensive tests for the three-state person tracker.

Covers:
    - Exponential decay math
    - Geometry helpers (IoU, face-inside-body)
    - Three-state transitions (Confirmed → Temporary → Unknown)
    - Head turn scenario (face lost, body tracked, face returns)
    - Occlusion scenario (person behind obstacle, body lost)
    - Path crossing scenario (two people swap positions)
    - Confidence decay over time
    - Face-to-body spatial mapping
"""

import math
import unittest
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from src.person_tracker import (
    AsyncBodyDetector,
    BodyDetector,
    IdentityState,
    PersonTracker,
    TrackedPerson,
    TrackingSignal,
    box_center,
    box_iou,
    exponential_decay,
    face_inside_body,
)
from src.recognize import MatchResult


# ─────────────────────────────────────────────────────────────
# Mock objects
# ─────────────────────────────────────────────────────────────


class KeypointsArray(np.ndarray):
    """Keypoints that carry a person_id tag for mock identity resolution."""

    def __new__(cls, input_array, person_id="unknown"):
        obj = np.asarray(input_array).view(cls)
        obj.person_id = person_id
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.person_id = getattr(obj, "person_id", "unknown")


class AlignedPatch(np.ndarray):
    """Aligned face patch that carries a person_id tag."""

    def __new__(cls, input_array, person_id="unknown"):
        obj = np.asarray(input_array).view(cls)
        obj.person_id = person_id
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.person_id = getattr(obj, "person_id", "unknown")


@dataclass
class MockFace:
    """Mock face detection with bounding box and tagged keypoints."""

    x1: int
    y1: int
    x2: int
    y2: int
    kps: np.ndarray
    person_id: str
    _cached_match: object = None


class MockFaceDetector:
    """Returns pre-configured face detections per frame."""

    def __init__(self, faces_per_frame: List[List[MockFace]]):
        self.faces_per_frame = faces_per_frame
        self.frame_idx = 0

    def detect(self, frame, max_faces=8):
        if self.frame_idx < len(self.faces_per_frame):
            faces = self.faces_per_frame[self.frame_idx]
            self.frame_idx += 1
            return faces
        return []


class MockBodyDetector:
    """Returns pre-configured body bounding boxes per frame."""

    def __init__(self, bodies_per_frame: List[List[Tuple[int, int, int, int]]]):
        self.bodies_per_frame = bodies_per_frame
        self.frame_idx = 0

    def detect(self, frame) -> List[Tuple[int, int, int, int]]:
        if self.frame_idx < len(self.bodies_per_frame):
            bodies = self.bodies_per_frame[self.frame_idx]
            self.frame_idx += 1
            return bodies
        return []


class MockEmbedder:
    """Returns deterministic embeddings based on person_id tag."""

    def embed(self, aligned_patch):
        pid = getattr(aligned_patch, "person_id", "unknown")
        if pid == "Alice":
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        elif pid == "Bob":
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)


class MockMatcher:
    """Matches embeddings against a mock database."""

    def __init__(self):
        self.db: Dict[str, np.ndarray] = {
            "Alice": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "Bob": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        }

    def match(self, emb):
        vec = np.asarray(emb, dtype=np.float32).reshape(-1)
        best_name = None
        best_sim = -1.0
        for name, ref in self.db.items():
            sim = float(np.dot(vec, ref))
            if sim > best_sim:
                best_sim = sim
                best_name = name

        if best_sim >= 0.5:
            return MatchResult(
                name=best_name,
                distance=1.0 - best_sim,
                similarity=best_sim,
                accepted=True,
            )
        return MatchResult(name=None, distance=1.0, similarity=0.0, accepted=False)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────


def _kps(person_id: str) -> KeypointsArray:
    """Create dummy 5-point keypoints tagged with a person_id."""
    pts = [[30, 30], [70, 30], [50, 50], [35, 70], [65, 70]]
    return KeypointsArray(pts, person_id=person_id)


def _face(x1, y1, x2, y2, person_id) -> MockFace:
    """Create a mock face detection."""
    return MockFace(
        x1=x1, y1=y1, x2=x2, y2=y2,
        kps=_kps(person_id),
        person_id=person_id,
    )


def _make_tracker(
    face_frames, body_frames, target="Alice", decay_rate=0.15, threshold=0.25,
    time_fn=None,
):
    """Create a PersonTracker with mock dependencies and monkey-patched align."""
    import src.person_tracker as pt

    original_identify = pt.PersonTracker._identify

    def mock_identify(self, frame, face):
        if hasattr(face, "_cached_match") and face._cached_match is not None:
            return face._cached_match
        pid = getattr(face, "person_id", None) or getattr(face.kps, "person_id", "unknown")
        emb = self.embedder.embed(
            AlignedPatch(np.zeros((112, 112, 3), dtype=np.uint8), person_id=pid)
        )
        match = self.matcher.match(emb)
        face._cached_match = match
        return match

    pt.PersonTracker._identify = mock_identify

    tracker = PersonTracker(
        target_name=target,
        face_detector=MockFaceDetector(face_frames),
        embedder=MockEmbedder(),
        matcher=MockMatcher(),
        body_detector=MockBodyDetector(body_frames),
        decay_rate=decay_rate,
        temporary_threshold=threshold,
        body_detect_interval=1,  # Detect bodies every frame in tests
        time_fn=time_fn,
    )

    # Restore original on cleanup
    tracker._orig_identify = original_identify
    tracker._module = pt
    return tracker


def _cleanup_tracker(tracker):
    """Restore the original _identify method after testing."""
    if hasattr(tracker, "_orig_identify") and hasattr(tracker, "_module"):
        tracker._module.PersonTracker._identify = tracker._orig_identify


# ─────────────────────────────────────────────────────────────
# Tests: Math / Geometry
# ─────────────────────────────────────────────────────────────


class TestExponentialDecay(unittest.TestCase):
    """Test the exponential confidence decay function."""

    def test_no_elapsed_time(self):
        """At t=0, confidence should equal initial value."""
        self.assertAlmostEqual(exponential_decay(1.0, 0.0, 0.15), 1.0)

    def test_decay_decreases_over_time(self):
        """Confidence should strictly decrease as time increases."""
        c1 = exponential_decay(1.0, 1.0, 0.15)
        c2 = exponential_decay(1.0, 5.0, 0.15)
        c3 = exponential_decay(1.0, 10.0, 0.15)
        self.assertGreater(c1, c2)
        self.assertGreater(c2, c3)

    def test_decay_follows_exponential_formula(self):
        """Values should match e^(-λt) precisely."""
        decay_rate = 0.2
        for t in [0.5, 1.0, 2.0, 5.0, 10.0]:
            expected = math.exp(-decay_rate * t)
            actual = exponential_decay(1.0, t, decay_rate)
            self.assertAlmostEqual(actual, expected, places=6)

    def test_high_decay_rate_fast_convergence(self):
        """High λ should make confidence drop rapidly."""
        c = exponential_decay(1.0, 5.0, 2.0)
        self.assertLess(c, 0.001)

    def test_low_decay_rate_slow_convergence(self):
        """Low λ should keep confidence high for longer."""
        c = exponential_decay(1.0, 5.0, 0.01)
        self.assertGreater(c, 0.9)

    def test_clamped_to_zero_one(self):
        """Result should always be in [0, 1]."""
        self.assertGreaterEqual(exponential_decay(1.0, 100.0, 10.0), 0.0)
        self.assertLessEqual(exponential_decay(1.0, 0.0, 0.0), 1.0)


class TestGeometryHelpers(unittest.TestCase):
    """Test IoU and face-inside-body spatial mapping."""

    def test_box_iou_identical(self):
        """Identical boxes should have IoU = 1.0."""
        box = (100, 100, 200, 200)
        self.assertAlmostEqual(box_iou(box, box), 1.0)

    def test_box_iou_no_overlap(self):
        """Non-overlapping boxes should have IoU = 0.0."""
        a = (0, 0, 50, 50)
        b = (100, 100, 200, 200)
        self.assertAlmostEqual(box_iou(a, b), 0.0)

    def test_box_iou_partial_overlap(self):
        """Partially overlapping boxes should have 0 < IoU < 1."""
        a = (0, 0, 100, 100)
        b = (50, 50, 150, 150)
        iou = box_iou(a, b)
        self.assertGreater(iou, 0.0)
        self.assertLess(iou, 1.0)

    def test_box_center(self):
        """Center should be the midpoint of XYXY coordinates."""
        c = box_center((100, 200, 300, 400))
        np.testing.assert_array_almost_equal(c, [200.0, 300.0])

    def test_face_inside_body_contained(self):
        """Face fully inside body should return True."""
        face = (150, 50, 250, 150)
        body = (100, 0, 300, 500)
        self.assertTrue(face_inside_body(face, body))

    def test_face_inside_body_outside(self):
        """Face clearly outside body should return False."""
        face = (500, 500, 600, 600)
        body = (100, 0, 300, 400)
        self.assertFalse(face_inside_body(face, body))

    def test_face_inside_body_with_margin(self):
        """Face slightly outside body but within margin should return True."""
        body = (100, 100, 200, 400)
        # Face just outside the body's right edge
        face = (195, 100, 215, 140)
        self.assertTrue(face_inside_body(face, body, margin=0.3))

    def test_face_inside_body_zero_margin(self):
        """With zero margin, face must be strictly inside."""
        body = (100, 100, 200, 400)
        face = (195, 100, 250, 140)  # Center at 222.5 — outside
        self.assertFalse(face_inside_body(face, body, margin=0.0))


# ─────────────────────────────────────────────────────────────
# Tests: Three-State Transitions
# ─────────────────────────────────────────────────────────────


class TestThreeStateModel(unittest.TestCase):
    """Test the three-state identity model transitions."""

    def test_unknown_to_confirmed_on_face_match(self):
        """State should go to CONFIRMED when target face is found."""
        face_frames = [
            [_face(100, 100, 200, 200, "Alice")],
        ]
        body_frames = [
            [(80, 50, 220, 450)],
        ]

        tracker = _make_tracker(face_frames, body_frames)
        try:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            person, signal = tracker.update(frame)

            self.assertIsNotNone(person)
            self.assertEqual(person.state, IdentityState.CONFIRMED)
            self.assertEqual(person.identity, "Alice")
            self.assertAlmostEqual(person.confidence, 1.0)
        finally:
            _cleanup_tracker(tracker)

    def test_confirmed_to_temporary_on_face_loss_body_present(self):
        """
        When face is lost but body is still tracked,
        state should transition to TEMPORARY with decaying confidence.
        """
        face_frames = [
            [_face(100, 100, 200, 200, "Alice")],  # Frame 0: face visible
            [],                                       # Frame 1: face hidden
        ]
        body_frames = [
            [(80, 50, 220, 450)],   # Frame 0: body visible
            [(80, 50, 220, 450)],   # Frame 1: body still visible
        ]

        t = [0.0]

        def mock_time():
            return t[0]

        tracker = _make_tracker(face_frames, body_frames, time_fn=mock_time)
        try:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

            # Frame 0: CONFIRMED
            t[0] = 0.0
            person, _ = tracker.update(frame)
            self.assertEqual(person.state, IdentityState.CONFIRMED)

            # Frame 1: face hidden, body present → TEMPORARY
            t[0] = 1.0  # 1 second elapsed
            person, signal = tracker.update(frame)
            self.assertIsNotNone(person)
            self.assertEqual(person.state, IdentityState.TEMPORARY)
            self.assertEqual(person.identity, "Alice")
            self.assertLess(person.confidence, 1.0)
            self.assertGreater(person.confidence, 0.0)
            # Signal should still be provided (tracking body)
            self.assertIsNotNone(signal)
        finally:
            _cleanup_tracker(tracker)

    def test_temporary_to_unknown_on_decay(self):
        """
        When confidence decays below threshold, state should go to UNKNOWN.
        """
        face_frames = [
            [_face(100, 100, 200, 200, "Alice")],
            [],
        ]
        body_frames = [
            [(80, 50, 220, 450)],
            [(80, 50, 220, 450)],
        ]

        t = [0.0]

        def mock_time():
            return t[0]

        # Use high decay rate so confidence drops quickly
        tracker = _make_tracker(
            face_frames, body_frames, decay_rate=2.0, threshold=0.25,
            time_fn=mock_time,
        )
        try:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

            # Frame 0: CONFIRMED
            t[0] = 0.0
            tracker.update(frame)

            # Frame 1: 5 seconds later with high decay → should drop below threshold
            t[0] = 5.0
            person, signal = tracker.update(frame)

            # Confidence e^(-2.0 * 5.0) ≈ 0.000045 — well below 0.25
            self.assertIsNone(person)
            self.assertEqual(tracker.state, IdentityState.UNKNOWN)
        finally:
            _cleanup_tracker(tracker)

    def test_temporary_to_confirmed_on_face_return(self):
        """
        When face returns after being in TEMPORARY, state should go back to CONFIRMED.
        """
        face_frames = [
            [_face(100, 100, 200, 200, "Alice")],  # Frame 0: face
            [],                                       # Frame 1: no face
            [_face(100, 100, 200, 200, "Alice")],  # Frame 2: face returns
        ]
        body_frames = [
            [(80, 50, 220, 450)],
            [(80, 50, 220, 450)],
            [(80, 50, 220, 450)],
        ]

        t = [0.0]

        def mock_time():
            return t[0]

        tracker = _make_tracker(face_frames, body_frames, time_fn=mock_time)
        try:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

            # Frame 0: CONFIRMED
            t[0] = 0.0
            tracker.update(frame)

            # Frame 1: TEMPORARY
            t[0] = 1.0
            person, _ = tracker.update(frame)
            self.assertEqual(person.state, IdentityState.TEMPORARY)

            # Frame 2: face returns → CONFIRMED
            t[0] = 2.0
            person, signal = tracker.update(frame)
            self.assertIsNotNone(person)
            self.assertEqual(person.state, IdentityState.CONFIRMED)
            self.assertAlmostEqual(person.confidence, 1.0)
        finally:
            _cleanup_tracker(tracker)


# ─────────────────────────────────────────────────────────────
# Tests: Real-World Scenarios
# ─────────────────────────────────────────────────────────────


class TestHeadTurnScenario(unittest.TestCase):
    """
    Simulate a head turn where the face disappears for several frames
    while the body remains visible, then the face comes back.
    """

    def test_head_turn_maintains_identity(self):
        face_frames = [
            [_face(100, 80, 200, 180, "Alice")],   # Looking at camera
            [_face(100, 80, 200, 180, "Alice")],   # Still looking
            [],                                       # Head turned — face gone
            [],                                       # Still turned
            [],                                       # Still turned
            [_face(100, 80, 200, 180, "Alice")],   # Turned back
        ]
        body_frames = [
            [(80, 50, 220, 450)],
            [(80, 50, 220, 450)],
            [(80, 50, 220, 450)],   # Body visible through turn
            [(82, 52, 222, 452)],   # Slight movement
            [(84, 54, 224, 454)],   # Slight movement
            [(80, 50, 220, 450)],
        ]

        t = [0.0]

        def mock_time():
            return t[0]

        tracker = _make_tracker(face_frames, body_frames, time_fn=mock_time)
        try:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

            # Frames 0-1: CONFIRMED
            for i in range(2):
                t[0] = i * 0.033
                person, _ = tracker.update(frame)
                self.assertEqual(person.state, IdentityState.CONFIRMED)

            # Frames 2-4: TEMPORARY (head turned, body visible)
            for i in range(2, 5):
                t[0] = i * 0.033
                person, signal = tracker.update(frame)
                self.assertIsNotNone(person)
                self.assertEqual(person.state, IdentityState.TEMPORARY)
                self.assertEqual(person.identity, "Alice")
                self.assertGreater(person.confidence, 0.0)
                # Body tracking should still provide a signal
                self.assertIsNotNone(signal)

            # Frame 5: face returns → CONFIRMED
            t[0] = 5 * 0.033
            person, _ = tracker.update(frame)
            self.assertEqual(person.state, IdentityState.CONFIRMED)
            self.assertAlmostEqual(person.confidence, 1.0)
        finally:
            _cleanup_tracker(tracker)


class TestOcclusionScenario(unittest.TestCase):
    """
    Simulate a person walking behind an obstacle where both face
    and body are lost temporarily.
    """

    def test_occlusion_drops_to_unknown(self):
        face_frames = [
            [_face(100, 100, 200, 200, "Alice")],  # Visible
            [],                                       # Behind obstacle
            [],                                       # Still hidden
        ]
        body_frames = [
            [(80, 50, 220, 450)],  # Visible
            [],                      # Behind obstacle — body also gone
            [],                      # Still hidden
        ]

        t = [0.0]

        def mock_time():
            return t[0]

        tracker = _make_tracker(
            face_frames, body_frames, decay_rate=1.0, threshold=0.25,
            time_fn=mock_time,
        )
        try:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

            # Frame 0: CONFIRMED
            t[0] = 0.0
            person, _ = tracker.update(frame)
            self.assertEqual(person.state, IdentityState.CONFIRMED)

            # Frame 1: Both lost — confidence starts decaying
            t[0] = 0.5
            person, signal = tracker.update(frame)
            # With decay_rate=1.0, t=0.5: e^(-0.5) ≈ 0.607 — still above 0.25
            self.assertIsNotNone(person)
            self.assertEqual(person.state, IdentityState.TEMPORARY)

            # Frame 2: Still hidden, more time passed
            t[0] = 2.0
            person, signal = tracker.update(frame)
            # e^(-2.0) ≈ 0.135 — below 0.25 → UNKNOWN
            self.assertIsNone(person)
            self.assertEqual(tracker.state, IdentityState.UNKNOWN)
        finally:
            _cleanup_tracker(tracker)


class TestPathCrossingScenario(unittest.TestCase):
    """
    Simulate two people crossing paths. The tracker should maintain
    the correct identity and not swap to the wrong person.
    """

    def test_path_crossing_no_identity_swap(self):
        """
        Alice and Bob cross paths. Alice's face is briefly lost.
        The tracker should NOT lock onto Bob.
        """
        # Alice is at x=100-200, Bob is at x=400-500
        # They cross: Alice moves right, Bob moves left
        face_frames = [
            [_face(100, 100, 200, 200, "Alice"), _face(400, 100, 500, 200, "Bob")],
            # During crossing: only Bob's face is visible (Alice turned)
            [_face(300, 100, 400, 200, "Bob")],
            # After crossing: Alice's face reappears at new position
            [_face(400, 100, 500, 200, "Alice"), _face(100, 100, 200, 200, "Bob")],
        ]
        body_frames = [
            [(80, 50, 220, 450), (380, 50, 520, 450)],       # Both visible
            [(200, 50, 340, 450), (260, 50, 420, 450)],       # Crossing
            [(380, 50, 520, 450), (80, 50, 220, 450)],        # Swapped positions
        ]

        t = [0.0]

        def mock_time():
            return t[0]

        tracker = _make_tracker(face_frames, body_frames, target="Alice", time_fn=mock_time)
        try:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

            # Frame 0: Lock on Alice
            t[0] = 0.0
            person, _ = tracker.update(frame)
            self.assertEqual(person.state, IdentityState.CONFIRMED)
            self.assertEqual(person.identity, "Alice")

            # Frame 1: Only Bob visible — tracker should NOT lock onto Bob
            t[0] = 0.033
            person, _ = tracker.update(frame)
            if person is not None:
                # If tracking body, should still be Alice in TEMPORARY
                self.assertEqual(person.identity, "Alice")
                self.assertNotEqual(person.identity, "Bob")

            # Frame 2: Alice reappears at new position
            t[0] = 0.066
            person, _ = tracker.update(frame)
            self.assertIsNotNone(person)
            self.assertEqual(person.identity, "Alice")
            self.assertEqual(person.state, IdentityState.CONFIRMED)
        finally:
            _cleanup_tracker(tracker)


class TestConfidenceDecayOverTime(unittest.TestCase):
    """Verify that confidence follows the exponential curve precisely."""

    def test_confidence_matches_exponential_formula(self):
        """
        Over multiple frames with face hidden, confidence should
        follow c = e^(-λt) within floating-point tolerance.
        """
        decay_rate = 0.15
        n_frames = 10

        face_frames = [[_face(100, 100, 200, 200, "Alice")]]
        face_frames += [[] for _ in range(n_frames)]

        body_frames = [[(80, 50, 220, 450)] for _ in range(n_frames + 1)]

        t = [0.0]

        def mock_time():
            return t[0]

        tracker = _make_tracker(
            face_frames, body_frames, decay_rate=decay_rate, time_fn=mock_time
        )
        try:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

            # Frame 0: confirm
            t[0] = 0.0
            tracker.update(frame)

            # Frames 1..n: check decay
            for i in range(1, n_frames + 1):
                elapsed = i * 0.5  # 0.5 second intervals
                t[0] = elapsed
                person, _ = tracker.update(frame)

                if person is None:
                    # Confidence dropped below threshold
                    expected = math.exp(-decay_rate * elapsed)
                    self.assertLess(expected, 0.25)
                    break

                expected = math.exp(-decay_rate * elapsed)
                self.assertAlmostEqual(person.confidence, expected, places=4)
        finally:
            _cleanup_tracker(tracker)


class TestFaceToBodyMapping(unittest.TestCase):
    """Test the spatial face-to-body mapping logic."""

    def test_face_mapped_to_correct_body(self):
        """Each face should be mapped to the body that contains it."""
        tracker = PersonTracker(
            target_name="Alice",
            face_detector=MockFaceDetector([]),
            embedder=MockEmbedder(),
            matcher=MockMatcher(),
            body_detector=MockBodyDetector([]),
        )

        faces = [
            _face(150, 50, 250, 150, "Alice"),   # Inside body 0
            _face(450, 50, 550, 150, "Bob"),      # Inside body 1
        ]
        bodies = [
            (100, 0, 300, 500),   # Body 0
            (400, 0, 600, 500),   # Body 1
        ]

        mapping = tracker._map_faces_to_bodies(faces, bodies)

        # Alice's face should be in body 0
        self.assertEqual(len(mapping[0]), 1)
        self.assertEqual(mapping[0][0].person_id, "Alice")

        # Bob's face should be in body 1
        self.assertEqual(len(mapping[1]), 1)
        self.assertEqual(mapping[1][0].person_id, "Bob")

    def test_unmatched_face_not_mapped(self):
        """A face outside all body boxes should not be mapped."""
        tracker = PersonTracker(
            target_name="Alice",
            face_detector=MockFaceDetector([]),
            embedder=MockEmbedder(),
            matcher=MockMatcher(),
            body_detector=MockBodyDetector([]),
        )

        faces = [
            _face(800, 50, 900, 150, "Alice"),  # Far outside body
        ]
        bodies = [
            (100, 0, 300, 500),
        ]

        mapping = tracker._map_faces_to_bodies(faces, bodies)
        self.assertEqual(len(mapping[0]), 0)


class TestPerformanceAndDetector(unittest.TestCase):
    """Test performance optimizations and detector behaviors."""

    def test_body_detector_init_and_detect_blank(self):
        """BodyDetector should initialize and return a list on a blank frame."""
        detector = BodyDetector()
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        boxes = detector.detect(img)
        self.assertIsInstance(boxes, list)

    def test_async_body_detector_lifecycle(self):
        """AsyncBodyDetector should accept frames and close cleanly."""
        detector = AsyncBodyDetector()
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        boxes = detector.detect(img)
        self.assertIsInstance(boxes, list)
        detector.close()

    def test_verify_every_skips_arcface_on_intermediate_frames(self):
        """Tracker should only call _identify on verify_every frames when locked."""
        identify_calls = [0]

        import src.person_tracker as pt
        orig_identify = pt.PersonTracker._identify

        def counted_identify(tracker_self, frame, face):
            identify_calls[0] += 1
            return MatchResult(name="Alice", distance=0.1, similarity=0.9, accepted=True)

        pt.PersonTracker._identify = counted_identify

        try:
            # 10 consecutive frames where Alice is stationary
            face_frames = [[_face(100, 100, 200, 200, "Alice")] for _ in range(10)]
            body_frames = [[(80, 50, 220, 450)] for _ in range(10)]

            tracker = PersonTracker(
                target_name="Alice",
                face_detector=MockFaceDetector(face_frames),
                embedder=MockEmbedder(),
                matcher=MockMatcher(),
                body_detector=MockBodyDetector(body_frames),
                verify_every=5,
            )

            frame = np.zeros((480, 640, 3), dtype=np.uint8)

            # Frame 1: acquisition (identify called 1 time)
            tracker.update(frame)
            self.assertEqual(tracker.state, IdentityState.CONFIRMED)
            self.assertEqual(identify_calls[0], 1)

            # Frames 2-4: spatial tracking, identify should NOT be called!
            for _ in range(3):
                tracker.update(frame)
                self.assertEqual(tracker.state, IdentityState.CONFIRMED)
            self.assertEqual(identify_calls[0], 1, "Intermediate frames should NOT call face recognition")

            # Frame 5: verify_every=5 hit -> identify called 2nd time
            tracker.update(frame)
            self.assertEqual(identify_calls[0], 2, "Verification frame should call face recognition")
        finally:
            pt.PersonTracker._identify = orig_identify


# ─────────────────────────────────────────────────────────────
# Tests: Body Feature Learning & 180° Re-Identification
# ─────────────────────────────────────────────────────────────


def _draw_test_body(frame, box, top_color=(255, 0, 0), bottom_color=(30, 30, 30)):
    """Draw a synthetic body with distinct clothing colors onto a frame."""
    x1, y1, x2, y2 = box
    h = y2 - y1
    # Head (0 to 15%)
    frame[y1: y1 + int(h * 0.15), x1:x2] = (180, 190, 210)
    # Upper torso / shirt (15% to 55%)
    frame[y1 + int(h * 0.15): y1 + int(h * 0.55), x1:x2] = top_color
    # Lower body / pants (55% to 92%)
    frame[y1 + int(h * 0.55): y1 + int(h * 0.92), x1:x2] = bottom_color
    # Shoes (92% to 100%)
    frame[y1 + int(h * 0.92):y2, x1:x2] = (40, 40, 40)


class TestBodyLearningAndReID(unittest.TestCase):
    """
    Test suite for whole-person body feature learning,
    180-degree rotation re-identification, and stranger rejection.
    """

    def test_180_rotation_identifies_target_via_body(self):
        """
        Target face confirmed for multiple frames -> Body features learned.
        Target turns 180° (face hidden from camera).
        Target body is recognized via learned appearance features.
        """
        # 8 frames with face confirmed, then 3 frames with face hidden (180° turn)
        face_frames = [[_face(110, 60, 190, 140, "Alice")] for _ in range(8)]
        face_frames += [[] for _ in range(3)]

        body_box = (80, 50, 220, 450)
        body_frames = [[body_box] for _ in range(11)]

        tracker = _make_tracker(face_frames, body_frames, target="Alice")
        try:
            # Frames 0..7: Alice facing camera, learning body features
            for i in range(8):
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                _draw_test_body(frame, body_box, top_color=(220, 30, 20), bottom_color=(30, 30, 30))
                person, _ = tracker.update(frame)
                self.assertIsNotNone(person)
                self.assertEqual(person.state, IdentityState.CONFIRMED)

            # Profile should now be ready
            self.assertTrue(tracker.target_profile.is_ready)
            self.assertEqual(tracker.target_profile.sample_count, 8)

            # Frames 8..10: Alice turns 180° (face hidden, back to camera)
            for i in range(3):
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                # Same clothing colors seen from the back, hair on head instead of face skin
                _draw_test_body(frame, body_box, top_color=(215, 32, 22), bottom_color=(32, 30, 32))
                frame[50: 50 + int(400 * 0.15), 80:220] = (25, 25, 25)  # dark hair

                person, signal = tracker.update(frame)

                self.assertIsNotNone(person, f"Failed to re-identify target on 180° turn frame {i}")
                self.assertEqual(person.identity, "Alice")
                self.assertEqual(person.state, IdentityState.TEMPORARY)
                self.assertTrue(person.is_reid, "Person should be marked as is_reid=True")
                self.assertGreaterEqual(person.body_sim, 0.70)
                self.assertIsNotNone(signal)
        finally:
            _cleanup_tracker(tracker)

    def test_stranger_facing_away_is_rejected(self):
        """
        Target profile learned.
        Target leaves and a Stranger enters facing away in different clothes.
        The Stranger must NOT be identified as the target (state UNKNOWN / rejected).
        """
        alice_box = (80, 50, 220, 450)
        stranger_box = (300, 50, 440, 450)

        face_frames = [[_face(110, 60, 190, 140, "Alice")] for _ in range(8)]
        # Frame 8: Alice is gone, stranger enters facing away (no face detected)
        face_frames.append([])

        body_frames = [[alice_box] for _ in range(8)]
        body_frames.append([stranger_box])

        tracker = _make_tracker(face_frames, body_frames, target="Alice")
        try:
            # Learn Alice's body (Red shirt, dark pants)
            for i in range(8):
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                _draw_test_body(frame, alice_box, top_color=(0, 0, 220), bottom_color=(20, 20, 20))
                tracker.update(frame)

            self.assertTrue(tracker.target_profile.is_ready)

            # Stranger enters (Yellow shirt, white pants, facing away)
            frame_stranger = np.zeros((480, 640, 3), dtype=np.uint8)
            _draw_test_body(frame_stranger, stranger_box, top_color=(0, 220, 220), bottom_color=(240, 240, 240))

            person, signal = tracker.update(frame_stranger)

            # Stranger must NOT be identified as Alice!
            # The stranger's body box must NEVER be assigned to Alice
            if person is not None:
                self.assertNotEqual(person.body_box, stranger_box, "Stranger was falsely identified as Alice!")
                self.assertIsNone(person.body_box)
                self.assertIsNone(signal, "No position signal should be generated for a stranger")
        finally:
            _cleanup_tracker(tracker)

    def test_target_and_stranger_both_facing_away(self):
        """
        Target profile learned.
        Both Target (facing away) and Stranger (facing away) are in the frame simultaneously.
        Tracker must match Target's body and ignore Stranger.
        """
        target_box = (80, 50, 220, 450)
        stranger_box = (350, 50, 490, 450)

        face_frames = [[_face(110, 60, 190, 140, "Alice")] for _ in range(8)]
        # Frame 8: Both Alice and Stranger facing away (no faces detected)
        face_frames.append([])

        body_frames = [[target_box] for _ in range(8)]
        body_frames.append([target_box, stranger_box])

        tracker = _make_tracker(face_frames, body_frames, target="Alice")
        try:
            # Learn Alice's body
            for i in range(8):
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                _draw_test_body(frame, target_box, top_color=(180, 100, 20), bottom_color=(120, 160, 200))
                tracker.update(frame)

            # Simultaneous frame: Target and Stranger both facing away
            frame_both = np.zeros((480, 640, 3), dtype=np.uint8)
            # Alice: Blue sweater, khaki pants
            _draw_test_body(frame_both, target_box, top_color=(175, 95, 25), bottom_color=(118, 158, 195))
            # Stranger: Green shirt, black pants
            _draw_test_body(frame_both, stranger_box, top_color=(20, 200, 20), bottom_color=(20, 20, 20))

            person, _ = tracker.update(frame_both)

            self.assertIsNotNone(person)
            self.assertEqual(person.identity, "Alice")
            self.assertEqual(person.body_box, target_box, "Tracker selected stranger box instead of target box!")
            self.assertTrue(person.is_reid)
            self.assertGreaterEqual(person.body_sim, 0.70)
        finally:
            _cleanup_tracker(tracker)


if __name__ == "__main__":
    unittest.main()

