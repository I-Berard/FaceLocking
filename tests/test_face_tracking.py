# tests/test_face_tracking.py

import unittest
from dataclasses import dataclass
from typing import List, Dict
import numpy as np

from src.face_tracking import LockedFaceTracker, LockState
from src.recognize import MatchResult


class AlignedPatch(np.ndarray):
    def __new__(cls, input_array, person_id="unknown"):
        obj = np.asarray(input_array).view(cls)
        obj.person_id = person_id
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.person_id = getattr(obj, "person_id", "unknown")


class KeypointsArray(np.ndarray):
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
    x1: int
    y1: int
    x2: int
    y2: int
    kps: np.ndarray
    person_id: str


class MockDetector:
    def __init__(self, faces_per_frame: List[List[MockFace]]):
        self.faces_per_frame = faces_per_frame
        self.frame_idx = 0

    def detect(self, frame, max_faces=8):
        if self.frame_idx < len(self.faces_per_frame):
            faces = self.faces_per_frame[self.frame_idx]
            self.frame_idx += 1
            return faces
        return []


class MockEmbedder:
    def embed(self, aligned_patch):
        meta = getattr(aligned_patch, "person_id", "unknown")
        if meta == "Alice":
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        elif meta == "Bob":
            return np.array([0.0, 1.0, 0.0], dtype=np.float32)
        else:
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)


class MockMatcher:
    def __init__(self, dist_thresh=0.81):
        self.dist_thresh = dist_thresh
        self.db: Dict[str, np.ndarray] = {
            "Alice": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "Bob": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        }

    def match(self, emb):
        vec = np.asarray(emb, dtype=np.float32).reshape(-1)
        sim_alice = float(np.dot(vec, self.db["Alice"]))
        sim_bob = float(np.dot(vec, self.db["Bob"]))

        if sim_alice >= (1.0 - self.dist_thresh):
            return MatchResult(name="Alice", distance=1.0 - sim_alice, similarity=sim_alice, accepted=True)
        elif sim_bob >= (1.0 - self.dist_thresh):
            return MatchResult(name="Bob", distance=1.0 - sim_bob, similarity=sim_bob, accepted=True)
        else:
            return MatchResult(name=None, distance=1.0, similarity=0.0, accepted=False)


def create_dummy_kps(person_id: str):
    pts = [[30, 30], [70, 30], [50, 50], [35, 70], [65, 70]]
    return KeypointsArray(pts, person_id=person_id)


class TestFaceTracking(unittest.TestCase):
    def test_locked_face_tracker_exclusive_focus(self):
        import src.face_tracking as ft

        original_align = ft.align_face_5pt

        def dummy_align(frame, kps, out_size=(112, 112)):
            pid = getattr(kps, "person_id", "unknown")
            patch = AlignedPatch(np.zeros((112, 112, 3), dtype=np.uint8), person_id=pid)
            return patch, None

        ft.align_face_5pt = dummy_align

        try:
            kps_alice = create_dummy_kps("Alice")
            kps_bob = create_dummy_kps("Bob")
            kps_charlie = create_dummy_kps("Charlie")

            face_alice = MockFace(x1=100, y1=100, x2=200, y2=200, kps=kps_alice, person_id="Alice")
            face_bob_close = MockFace(x1=110, y1=110, x2=210, y2=210, kps=kps_bob, person_id="Bob")
            face_charlie = MockFace(x1=400, y1=100, x2=500, y2=200, kps=kps_charlie, person_id="Charlie")

            # Frames sequence:
            # 0: Only Alice -> Lock acquired on Alice
            # 1: Alice + Bob (Bob closer to Alice's last box!) -> Retains lock on Alice
            # 2: Alice leaves frame, only Bob remains -> Returns None, state LOST (does NOT lock onto Bob!)
            # 3: Alice returns -> Re-acquires lock on Alice
            frames_data = [
                [face_alice],
                [face_bob_close, face_alice],
                [face_bob_close, face_charlie],
                [face_alice],
            ]

            detector = MockDetector(frames_data)
            embedder = MockEmbedder()
            matcher = MockMatcher()

            tracker = LockedFaceTracker(
                target_name="Alice",
                detector=detector,
                embedder=embedder,
                matcher=matcher,
                verify_every=1,
            )

            frame_img = np.zeros((480, 640, 3), dtype=np.uint8)

            # Frame 0: Lock on Alice
            cand, signal = tracker.update(frame_img)
            self.assertIsNotNone(cand)
            self.assertEqual(cand.person_id, "Alice")
            self.assertEqual(tracker.state, LockState.LOCKED)

            # Frame 1: Bob enters close to Alice's box. Tracker MUST retain Alice!
            cand, signal = tracker.update(frame_img)
            self.assertIsNotNone(cand)
            self.assertEqual(cand.person_id, "Alice")
            self.assertEqual(tracker.state, LockState.LOCKED)

            # Frame 2: Alice leaves. Only Bob remains. Tracker MUST NOT lock onto Bob!
            cand, signal = tracker.update(frame_img)
            self.assertIsNone(cand)
            self.assertIsNone(signal)
            self.assertIn(tracker.state, (LockState.LOST, LockState.SEARCHING))

            # Frame 3: Alice returns. Tracker MUST re-acquire Alice!
            cand, signal = tracker.update(frame_img)
            self.assertIsNotNone(cand)
            self.assertEqual(cand.person_id, "Alice")
            self.assertEqual(tracker.state, LockState.LOCKED)

        finally:
            ft.align_face_5pt = original_align


if __name__ == "__main__":
    unittest.main()
