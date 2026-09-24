# src/recognize.py

"""
Multi-face face recognition using the 5-point pipeline:

Haar face detection
    -> MediaPipe FaceLandmarker 5-point landmarks per ROI
    -> ArcFace 5-point alignment to 112x112
    -> ArcFace ONNX embedding
    -> cosine distance against enrolled database
    -> identity label

Run:
    python -m src.recognize

Keys:
    q       Quit
    r       Reload database from disk
    +/-     Adjust recognition distance threshold
    d       Toggle debug overlay

Notes:
    - Haar detects multiple faces.
    - MediaPipe FaceLandmarker runs separately on each Haar ROI.
    - Running on each ROI improves landmark consistency with the Haar box.
    - The database is expected at:
          data/db/face_db.npz
    - Cosine distance:
          distance = 1 - cosine_similarity
    - Embeddings are L2-normalized, so:
          cosine_similarity = dot(a, b)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import mediapipe as mp

from .haar_5pt import align_face_5pt


# ---------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------


@dataclass
class FaceDet:
    """Detected face with bounding box and five facial landmarks."""

    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    kps: np.ndarray  # Shape: (5, 2), full-frame coordinates


@dataclass
class MatchResult:
    """Result of comparing an embedding against the face database."""

    name: Optional[str]
    distance: float
    similarity: float
    accepted: bool


# ---------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity between two vectors."""

    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)

    return float(np.dot(a, b))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine distance between two vectors."""

    return 1.0 - cosine_similarity(a, b)


# ---------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------


def _clip_xyxy(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    """Clip an XYXY bounding box to image boundaries."""

    x1 = int(max(0, min(width - 1, round(x1))))
    y1 = int(max(0, min(height - 1, round(y1))))
    x2 = int(max(0, min(width - 1, round(x2))))
    y2 = int(max(0, min(height - 1, round(y2))))

    if x2 < x1:
        x1, x2 = x2, x1

    if y2 < y1:
        y1, y2 = y2, y1

    return x1, y1, x2, y2


def _bbox_from_5pt(
    kps: np.ndarray,
    pad_x: float = 0.55,
    pad_y_top: float = 0.85,
    pad_y_bot: float = 1.15,
) -> np.ndarray:
    """
    Build a face-like bounding box from five landmarks.

    Args:
        kps: Five points with shape (5, 2), in full-frame coordinates.
        pad_x: Horizontal padding factor.
        pad_y_top: Padding above landmarks.
        pad_y_bot: Padding below landmarks.

    Returns:
        XYXY bounding box as float32.
    """

    k = kps.astype(np.float32)

    x_min = float(np.min(k[:, 0]))
    x_max = float(np.max(k[:, 0]))
    y_min = float(np.min(k[:, 1]))
    y_max = float(np.max(k[:, 1]))

    width = max(1.0, x_max - x_min)
    height = max(1.0, y_max - y_min)

    x1 = x_min - pad_x * width
    x2 = x_max + pad_x * width
    y1 = y_min - pad_y_top * height
    y2 = y_max + pad_y_bot * height

    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _kps_span_ok(
    kps: np.ndarray,
    min_eye_dist: float,
) -> bool:
    """
    Basic landmark geometry sanity check.

    Checks:
        - Eyes are not collapsed.
        - Mouth points are below the nose.
    """

    k = kps.astype(np.float32)

    left_eye, right_eye, nose, mouth_left, mouth_right = k

    eye_distance = float(np.linalg.norm(right_eye - left_eye))

    if eye_distance < float(min_eye_dist):
        return False

    if not (mouth_left[1] > nose[1] and mouth_right[1] > nose[1]):
        return False

    return True


# ---------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------


def load_db_npz(db_path: Path) -> Dict[str, np.ndarray]:
    """
    Load enrolled face embeddings from an NPZ database.

    Expected format:
        key   -> person name
        value -> embedding vector
    """

    if not db_path.exists():
        return {}

    data = np.load(str(db_path), allow_pickle=True)

    result: Dict[str, np.ndarray] = {}

    try:
        for key in data.files:
            result[key] = (
                np.asarray(data[key], dtype=np.float32)
                .reshape(-1)
            )
    finally:
        data.close()

    return result


# ---------------------------------------------------------------------
# Multi-face Haar + MediaPipe FaceLandmarker
# ---------------------------------------------------------------------


class HaarFaceLandmarker5pt:
    """
    Multi-face detector.

    Pipeline:
        Haar -> ROI -> MediaPipe FaceLandmarker -> five landmarks
    """

    IDX_LEFT_EYE = 33
    IDX_RIGHT_EYE = 263
    IDX_NOSE_TIP = 1
    IDX_MOUTH_LEFT = 61
    IDX_MOUTH_RIGHT = 291

    def __init__(
        self,
        haar_xml: Optional[str] = None,
        landmark_model: str = "models/face_landmarker.task",
        min_size: Tuple[int, int] = (70, 70),
        debug: bool = False,
    ) -> None:
        self.debug = bool(debug)
        self.min_size = tuple(map(int, min_size))

        # -------------------------------------------------------------
        # Haar detector
        # -------------------------------------------------------------

        if haar_xml is None:
            haar_xml = (
                "models/"
                "haarcascade_frontalface_default.xml"
            )

        self.face_cascade = cv2.CascadeClassifier(haar_xml)

        if self.face_cascade.empty():
            raise RuntimeError(
                f"Failed to load Haar cascade: {haar_xml}"
            )

        # -------------------------------------------------------------
        # MediaPipe FaceLandmarker
        # -------------------------------------------------------------

        model_path = Path(landmark_model)

        if not model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe FaceLandmarker model not found: "
                f"{model_path}"
            )

        base_options = mp.tasks.BaseOptions(
            model_asset_path=str(model_path)
        )

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=5,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )

        self.face_landmarker = (
            mp.tasks.vision.FaceLandmarker.create_from_options(
                options
            )
        )

        self._timestamp_ms = 0

    # -----------------------------------------------------------------
    # Haar
    # -----------------------------------------------------------------

    def _haar_faces(self, gray: np.ndarray) -> np.ndarray:
        """Detect faces with Haar cascade."""

        if not hasattr(self, "face_cascade") or self.face_cascade.empty():
            return np.zeros((0, 4), dtype=np.int32)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            flags=cv2.CASCADE_SCALE_IMAGE,
            minSize=self.min_size,
        )

        if faces is None or len(faces) == 0:
            return np.zeros((0, 4), dtype=np.int32)

        return faces.astype(np.int32)

    # -----------------------------------------------------------------
    # MediaPipe 5-point landmarks
    # -----------------------------------------------------------------

    def _full_frame_landmarks_5pt(
        self,
        frame_bgr: np.ndarray,
    ) -> List[np.ndarray]:
        """
        Run MediaPipe FaceLandmarker on full frame.

        Returns five points in full-frame coordinates for each detected face.
        """

        height, width = frame_bgr.shape[:2]

        frame_rgb = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2RGB,
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb,
        )

        self._timestamp_ms += 1

        result = self.face_landmarker.detect_for_video(
            mp_image,
            self._timestamp_ms,
        )

        if not result.face_landmarks:
            return []

        all_kps = []
        indices = [
            self.IDX_LEFT_EYE,
            self.IDX_RIGHT_EYE,
            self.IDX_NOSE_TIP,
            self.IDX_MOUTH_LEFT,
            self.IDX_MOUTH_RIGHT,
        ]

        for landmarks in result.face_landmarks:
            points = []
            for index in indices:
                landmark = landmarks[index]
                points.append(
                    [
                        landmark.x * width,
                        landmark.y * height,
                    ]
                )

            kps = np.asarray(points, dtype=np.float32)

            # Enforce consistent left/right ordering.
            if kps[0, 0] > kps[1, 0]:
                kps[[0, 1]] = kps[[1, 0]]

            if kps[3, 0] > kps[4, 0]:
                kps[[3, 4]] = kps[[4, 3]]

            all_kps.append(kps)

        return all_kps

    # -----------------------------------------------------------------
    # Public detection
    # -----------------------------------------------------------------

    def detect(
        self,
        frame_bgr: np.ndarray,
        max_faces: int = 5,
    ) -> List[FaceDet]:
        """Detect and return up to max_faces faces using MediaPipe full frame."""

        height, width = frame_bgr.shape[:2]

        # 1. Primary: MediaPipe full frame
        landmarks_list = self._full_frame_landmarks_5pt(frame_bgr)

        detections: List[FaceDet] = []

        if landmarks_list:
            for kps in landmarks_list:
                if not _kps_span_ok(kps, min_eye_dist=10.0):
                    continue

                bbox = _bbox_from_5pt(
                    kps,
                    pad_x=0.55,
                    pad_y_top=0.85,
                    pad_y_bot=1.15,
                )

                x1, y1, x2, y2 = _clip_xyxy(
                    bbox[0],
                    bbox[1],
                    bbox[2],
                    bbox[3],
                    width,
                    height,
                )

                detections.append(
                    FaceDet(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        score=1.0,
                        kps=kps.astype(np.float32),
                    )
                )

        # 2. Fallback: Haar cascade if MediaPipe returned no faces
        if not detections and hasattr(self, "face_cascade") and not self.face_cascade.empty():
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = self._haar_faces(gray)
            if faces.shape[0] > 0:
                areas = faces[:, 2] * faces[:, 3]
                order = np.argsort(areas)[::-1]
                faces = faces[order][:max_faces]

                for x, y, w, h in faces:
                    kps = np.array(
                        [
                            [x + 0.3 * w, y + 0.35 * h],
                            [x + 0.7 * w, y + 0.35 * h],
                            [x + 0.5 * w, y + 0.55 * h],
                            [x + 0.35 * w, y + 0.75 * h],
                            [x + 0.65 * w, y + 0.75 * h],
                        ],
                        dtype=np.float32,
                    )
                    x1, y1, x2, y2 = _clip_xyxy(x, y, x + w, y + h, width, height)
                    detections.append(
                        FaceDet(
                            x1=x1,
                            y1=y1,
                            x2=x2,
                            y2=y2,
                            score=0.8,
                            kps=kps,
                        )
                    )

        # Sort largest face first
        if detections:
            detections.sort(
                key=lambda d: (d.x2 - d.x1) * (d.y2 - d.y1),
                reverse=True,
            )

        return detections[:max_faces]

    def close(self) -> None:
        """Release MediaPipe resources."""

        if hasattr(self, "face_landmarker"):
            self.face_landmarker.close()


# ---------------------------------------------------------------------
# Face database matcher
# ---------------------------------------------------------------------


class FaceDBMatcher:
    """Match normalized embeddings against the enrolled database."""

    def __init__(
        self,
        db: Dict[str, np.ndarray],
        dist_thresh: float = 0.81,
    ) -> None:
        self.db = db
        self.dist_thresh = float(dist_thresh)

        self._names: List[str] = []
        self._mat: Optional[np.ndarray] = None

        self._rebuild()

    def _rebuild(self) -> None:
        """Build a matrix containing all enrolled embeddings."""

        self._names = sorted(self.db.keys())

        if self._names:
            self._mat = np.stack(
                [
                    self.db[name]
                    .reshape(-1)
                    .astype(np.float32)
                    for name in self._names
                ],
                axis=0,
            )
        else:
            self._mat = None

    def reload_from(self, path: Path) -> None:
        """Reload the database from disk."""

        self.db = load_db_npz(path)
        self._rebuild()

    def match(self, emb: Any) -> MatchResult:
        """Return the closest identity."""

        if self._mat is None or not self._names:
            return MatchResult(
                name=None,
                distance=1.0,
                similarity=0.0,
                accepted=False,
            )

        if hasattr(emb, "embedding"):
            emb = emb.embedding

        embedding = (
            np.asarray(emb, dtype=np.float32)
            .reshape(1, -1)
        )

        # Since embeddings are normalized:
        # cosine similarity = dot product.
        similarities = (
            self._mat @ embedding.T
        ).reshape(-1)

        best_index = int(
            np.argmax(similarities)
        )

        best_similarity = float(
            similarities[best_index]
        )

        best_distance = 1.0 - best_similarity

        accepted = (
            best_distance <= self.dist_thresh
        )

        return MatchResult(
            name=(
                self._names[best_index]
                if accepted
                else None
            ),
            distance=float(best_distance),
            similarity=float(best_similarity),
            accepted=bool(accepted),
        )


# ---------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------


def main() -> None:
    db_path = Path(
        "data/db/face_db.npz"
    )

    detector = HaarFaceLandmarker5pt(
        haar_xml=(
            "models/"
            "haarcascade_frontalface_default.xml"
        ),
        landmark_model=(
            "models/face_landmarker.task"
        ),
        min_size=(70, 70),
        debug=False,
    )

    try:
        from .embed import ArcFaceEmbedderONNX

        embedder = ArcFaceEmbedderONNX(
            model_path=(
                "models/embedder_arcface.onnx"
            ),
            input_size=(112, 112),
            debug=False,
        )

        db = load_db_npz(db_path)

        matcher = FaceDBMatcher(
            db=db,
            dist_thresh=0.81,
        )

        cap = None
        for cam_idx in [1, 0, 2]:
            temp_cap = cv2.VideoCapture(cam_idx)
            if temp_cap.isOpened():
                cap = temp_cap
                break
            temp_cap.release()

        if cap is None or not cap.isOpened():
            raise RuntimeError("Camera not available. Checked indices 1, 0, 2.")

        print(
            "Recognize (multi-face)"
        )
        print(
            "q=quit | r=reload DB | "
            "+/-=threshold | d=debug"
        )

        t0 = time.time()
        frames = 0
        fps: Optional[float] = None
        show_debug = False

        while True:
            ok, frame = cap.read()

            if not ok:
                print(
                    "[recognize] Failed to read "
                    "camera frame."
                )
                break

            faces = detector.detect(
                frame,
                max_faces=5,
            )

            vis = frame.copy()

            # ---------------------------------------------------------
            # FPS
            # ---------------------------------------------------------

            frames += 1

            dt = time.time() - t0

            if dt >= 1.0:
                fps = frames / dt
                frames = 0
                t0 = time.time()

            # ---------------------------------------------------------
            # Thumbnail layout
            # ---------------------------------------------------------

            height, width = vis.shape[:2]

            thumb = 112
            pad = 8

            x0 = width - thumb - pad
            y0 = 80

            shown = 0

            # ---------------------------------------------------------
            # Recognize every detected face
            # ---------------------------------------------------------

            for i, face in enumerate(faces):
                # Draw bounding box.
                cv2.rectangle(
                    vis,
                    (face.x1, face.y1),
                    (face.x2, face.y2),
                    (0, 255, 0),
                    2,
                )

                # Draw five landmarks.
                for x, y in face.kps.astype(int):
                    cv2.circle(
                        vis,
                        (int(x), int(y)),
                        2,
                        (0, 255, 0),
                        -1,
                    )

                # -----------------------------------------------------
                # Alignment
                # -----------------------------------------------------

                aligned, _ = align_face_5pt(
                    frame,
                    face.kps,
                    out_size=(112, 112),
                )

                # -----------------------------------------------------
                # Embedding
                # -----------------------------------------------------

                embedding = embedder.embed(
                    aligned
                )

                # -----------------------------------------------------
                # Matching
                # -----------------------------------------------------

                match = matcher.match(
                    embedding
                )

                label = (
                    match.name
                    if match.name is not None
                    else "Unknown"
                )

                line1 = label
                line2 = (
                    f"dist={match.distance:.3f} "
                    f"sim={match.similarity:.3f}"
                )

                # Known = green, unknown = red.
                color = (
                    (0, 255, 0)
                    if match.accepted
                    else (0, 0, 255)
                )

                # -----------------------------------------------------
                # Draw label
                # -----------------------------------------------------

                cv2.putText(
                    vis,
                    line1,
                    (
                        face.x1,
                        max(0, face.y1 - 28),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                )

                cv2.putText(
                    vis,
                    line2,
                    (
                        face.x1,
                        max(0, face.y1 - 6),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

                # -----------------------------------------------------
                # Aligned preview
                # -----------------------------------------------------

                if (
                    y0 + thumb <= height
                    and shown < 4
                ):
                    vis[
                        y0:y0 + thumb,
                        x0:x0 + thumb,
                    ] = aligned

                    cv2.putText(
                        vis,
                        f"{i + 1}:{label}",
                        (x0, y0 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                    )

                    y0 += thumb + pad
                    shown += 1

                # -----------------------------------------------------
                # Debug overlay
                # -----------------------------------------------------

                if show_debug:
                    debug_text = (
                        "kpsLeye="
                        f"({face.kps[0, 0]:.0f},"
                        f"{face.kps[0, 1]:.0f})"
                    )

                    cv2.putText(
                        vis,
                        debug_text,
                        (10, height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

            # ---------------------------------------------------------
            # Header
            # ---------------------------------------------------------

            header = (
                f"IDs={len(matcher._names)} "
                f"thr(dist)={matcher.dist_thresh:.2f}"
            )

            if fps is not None:
                header += f" fps={fps:.1f}"

            cv2.putText(
                vis,
                header,
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2,
            )

            cv2.imshow(
                "recognize",
                vis,
            )

            key = cv2.waitKey(1) & 0xFF

            # ---------------------------------------------------------
            # Controls
            # ---------------------------------------------------------

            if key == ord("q"):
                break

            elif key == ord("r"):
                matcher.reload_from(
                    db_path
                )

                print(
                    "[recognize] reloaded DB: "
                    f"{len(matcher._names)} identities"
                )

            elif key in (
                ord("+"),
                ord("="),
            ):
                matcher.dist_thresh = float(
                    min(
                        1.20,
                        matcher.dist_thresh + 0.01,
                    )
                )

                print(
                    "[recognize] "
                    f"thr(dist)={matcher.dist_thresh:.2f} "
                    f"(sim~{1.0 - matcher.dist_thresh:.2f})"
                )

            elif key == ord("-"):
                matcher.dist_thresh = float(
                    max(
                        0.05,
                        matcher.dist_thresh - 0.01,
                    )
                )

                print(
                    "[recognize] "
                    f"thr(dist)={matcher.dist_thresh:.2f} "
                    f"(sim~{1.0 - matcher.dist_thresh:.2f})"
                )

            elif key == ord("d"):
                show_debug = not show_debug

                print(
                    "[recognize] debug overlay: "
                    f"{'ON' if show_debug else 'OFF'}"
                )

        cap.release()
        cv2.destroyAllWindows()

    finally:
        detector.close()


if __name__ == "__main__":
    main()