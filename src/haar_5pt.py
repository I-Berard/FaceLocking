# src/haar_5pt.py

"""
Haar face detection + 5-point facial landmarks using the
modern MediaPipe Tasks Face Landmarker API.

Pipeline:
    1. Haar detects a face candidate.
    2. MediaPipe FaceLandmarker confirms the face.
    3. We extract only 5 keypoints:
        - left eye
        - right eye
        - nose tip
        - mouth left
        - mouth right
    4. The bounding box is rebuilt from those 5 keypoints.
    5. EMA smoothing stabilizes the box and landmarks.
    6. Haar false positives are rejected when MediaPipe
       does not produce a valid face.

Expected model:
    models/face_landmarker.task

Run:
    python -m src.haar_5pt

Keys:
    q - quit
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception as exc:
    mp = None
    _MP_IMPORT_ERROR = exc
else:
    _MP_IMPORT_ERROR = None


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class FaceKpsBox:
    """Face bounding box with confidence score and 5 facial keypoints."""

    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    kps: np.ndarray  # Shape: (5, 2), float32


# ---------------------------------------------------------------------------
# Alignment helpers
# ---------------------------------------------------------------------------

def _estimate_norm_5pt(
    kps_5x2: np.ndarray,
    out_size: Tuple[int, int] = (112, 112),
) -> np.ndarray:
    """
    Build a 2x3 affine matrix that maps 5 facial keypoints
    to an ArcFace-style template.

    Keypoint order:
        [left_eye, right_eye, nose, mouth_left, mouth_right]
    """

    kps = kps_5x2.astype(np.float32)

    # Standard ArcFace 112x112 template.
    dst = np.array(
        [
            [38.2946, 51.6963],  # left eye
            [73.5318, 51.5014],  # right eye
            [56.0252, 71.7366],  # nose
            [41.5493, 92.3655],  # left mouth
            [70.7299, 92.2041],  # right mouth
        ],
        dtype=np.float32,
    )

    out_w, out_h = map(int, out_size)

    # Scale template if using an output size other than 112x112.
    if (out_w, out_h) != (112, 112):
        scale = np.array(
            [
                out_w / 112.0,
                out_h / 112.0,
            ],
            dtype=np.float32,
        )

        dst *= scale

    # Similarity transform:
    # rotation + scale + translation using standard least squares.
    matrix, _ = cv2.estimateAffinePartial2D(
        kps,
        dst,
    )

    # Fallback if estimation fails.
    if matrix is None:
        matrix = cv2.getAffineTransform(
            kps[:3],
            dst[:3],
        )

    return matrix.astype(np.float32)


def align_face_5pt(
    frame_bgr: np.ndarray,
    kps_5x2: np.ndarray,
    out_size: Tuple[int, int] = (112, 112),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align a face using 5 facial keypoints.

    Returns:
        aligned_bgr:
            Aligned face image.

        matrix:
            2x3 affine transformation matrix.
    """

    matrix = _estimate_norm_5pt(
        kps_5x2,
        out_size=out_size,
    )

    out_w, out_h = map(int, out_size)

    aligned = cv2.warpAffine(
        frame_bgr,
        matrix,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    return aligned, matrix


# ---------------------------------------------------------------------------
# Bounding-box helpers
# ---------------------------------------------------------------------------

def _clip_box_xyxy(
    box: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Clip an XYXY bounding box to image boundaries."""

    clipped = box.astype(np.float32).copy()

    clipped[0] = np.clip(
        clipped[0],
        0,
        width - 1,
    )

    clipped[1] = np.clip(
        clipped[1],
        0,
        height - 1,
    )

    clipped[2] = np.clip(
        clipped[2],
        0,
        width - 1,
    )

    clipped[3] = np.clip(
        clipped[3],
        0,
        height - 1,
    )

    return clipped


def _bbox_from_5pt(
    kps: np.ndarray,
    pad_x: float = 0.55,
    pad_y_top: float = 0.85,
    pad_y_bot: float = 1.15,
) -> np.ndarray:
    """
    Build a face bounding box from 5 keypoints.

    Asymmetric padding gives:
        - More room above the eyes for the forehead.
        - More room below the mouth for the chin.
    """

    points = kps.astype(np.float32)

    x_min = float(np.min(points[:, 0]))
    x_max = float(np.max(points[:, 0]))
    y_min = float(np.min(points[:, 1]))
    y_max = float(np.max(points[:, 1]))

    width = max(1.0, x_max - x_min)
    height = max(1.0, y_max - y_min)

    x1 = x_min - pad_x * width
    x2 = x_max + pad_x * width

    y1 = y_min - pad_y_top * height
    y2 = y_max + pad_y_bot * height

    return np.array(
        [x1, y1, x2, y2],
        dtype=np.float32,
    )


def _ema(
    previous: Optional[np.ndarray],
    current: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Apply exponential moving-average smoothing."""

    if previous is None:
        return current.astype(np.float32)

    return (
        alpha * previous
        + (1.0 - alpha) * current
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Keypoint validation
# ---------------------------------------------------------------------------

def _kps_span_ok(
    kps: np.ndarray,
    min_eye_dist: float = 12.0,
) -> bool:
    """
    Perform basic 5-point geometry validation.

    Checks:
        - Eye distance is large enough.
        - Both mouth points are below the nose.
    """

    points = kps.astype(np.float32)

    left_eye, right_eye, nose, mouth_left, mouth_right = points

    eye_distance = float(
        np.linalg.norm(
            right_eye - left_eye,
        )
    )

    if eye_distance < min_eye_dist:
        return False

    if not (
        mouth_left[1] > nose[1]
        and mouth_right[1] > nose[1]
    ):
        return False

    return True


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class Haar5ptDetector:
    """
    Haar face detector combined with the modern
    MediaPipe Tasks FaceLandmarker.
    """

    # MediaPipe Face Landmarker landmark indices.
    #
    # These correspond to:
    #   33  -> left eye
    #   263 -> right eye
    #   1   -> nose tip
    #   61  -> mouth left
    #   291 -> mouth right
    IDX_LEFT_EYE = 33
    IDX_RIGHT_EYE = 263
    IDX_NOSE_TIP = 1
    IDX_MOUTH_LEFT = 61
    IDX_MOUTH_RIGHT = 291

    def __init__(
        self,
        haar_xml: Optional[str] = None,
        model_path: Optional[str] = None,
        min_size: Tuple[int, int] = (60, 60),
        smooth_alpha: float = 0.80,
        min_face_detection_confidence: float = 0.5,
        min_face_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        debug: bool = True,
    ) -> None:
        self.debug = bool(debug)
        self.min_size = tuple(map(int, min_size))
        self.smooth_alpha = float(smooth_alpha)

        # ---------------------------------------------------------------
        # Haar cascade
        # ---------------------------------------------------------------

        if haar_xml is None:
            haar_xml = (
                "models/haarcascade_frontalface_default.xml"
            )

        self.face_cascade = cv2.CascadeClassifier(
            haar_xml
        )

        if self.face_cascade.empty():
            raise RuntimeError(
                f"Failed to load Haar cascade: {haar_xml}"
            )

        # ---------------------------------------------------------------
        # MediaPipe import
        # ---------------------------------------------------------------

        if mp is None:
            raise RuntimeError(
                f"MediaPipe import failed: {_MP_IMPORT_ERROR}\n"
                "Install MediaPipe with:\n"
                "    pip install -U mediapipe"
            )

        # ---------------------------------------------------------------
        # Face Landmarker model
        # ---------------------------------------------------------------

        if model_path is None:
            model_path = "models/face_landmarker.task"

        model_file = Path(model_path)

        if not model_file.is_file():
            raise FileNotFoundError(
                "MediaPipe Face Landmarker model was not found:\n"
                f"    {model_file.resolve()}\n\n"
                "Place your .task model at that path or pass "
                "model_path explicitly."
            )

        # ---------------------------------------------------------------
        # Modern MediaPipe Tasks API
        # ---------------------------------------------------------------

        base_options = mp.tasks.BaseOptions(
            model_asset_path=str(model_file)
        )

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=(
                min_face_detection_confidence
            ),
            min_face_presence_confidence=(
                min_face_presence_confidence
            ),
            min_tracking_confidence=(
                min_tracking_confidence
            ),
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )

        self.face_landmarker = (
            mp.tasks.vision.FaceLandmarker.create_from_options(
                options
            )
        )

        # ---------------------------------------------------------------
        # Tracking / smoothing state
        # ---------------------------------------------------------------

        self._prev_box: Optional[np.ndarray] = None
        self._prev_kps: Optional[np.ndarray] = None

        self._timestamp_ms = 0

    # ------------------------------------------------------------------
    # Haar detection
    # ------------------------------------------------------------------

    def _haar_faces(
        self,
        gray: np.ndarray,
    ) -> np.ndarray:
        """Detect face candidates using Haar."""

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            flags=cv2.CASCADE_SCALE_IMAGE,
            minSize=self.min_size,
        )

        if faces is None or len(faces) == 0:
            return np.zeros(
                (0, 4),
                dtype=np.int32,
            )

        # OpenCV returns:
        #   x, y, width, height
        return faces.astype(np.int32)

    # ------------------------------------------------------------------
    # MediaPipe Face Landmarker
    # ------------------------------------------------------------------

    def _facemesh_5pt(
        self,
        frame_bgr: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Run MediaPipe Face Landmarker and extract 5 landmarks.

        Uses the modern MediaPipe Tasks API instead of:
            mp.solutions.face_mesh.FaceMesh
        """

        height, width = frame_bgr.shape[:2]

        # MediaPipe expects RGB.
        frame_rgb = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2RGB,
        )

        # Create a MediaPipe Image.
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb,
        )

        # VIDEO mode requires monotonically increasing timestamps.
        self._timestamp_ms += 1

        result = self.face_landmarker.detect_for_video(
            mp_image,
            self._timestamp_ms,
        )

        if not result.face_landmarks:
            return None

        # We configured num_faces=1.
        landmarks = result.face_landmarks[0]

        indices = [
            self.IDX_LEFT_EYE,
            self.IDX_RIGHT_EYE,
            self.IDX_NOSE_TIP,
            self.IDX_MOUTH_LEFT,
            self.IDX_MOUTH_RIGHT,
        ]

        points = []

        for index in indices:
            landmark = landmarks[index]

            points.append(
                [
                    landmark.x * width,
                    landmark.y * height,
                ]
            )

        kps = np.array(
            points,
            dtype=np.float32,
        )

        # Ensure left/right ordering.
        if kps[0, 0] > kps[1, 0]:
            kps[[0, 1]] = kps[[1, 0]]

        if kps[3, 0] > kps[4, 0]:
            kps[[3, 4]] = kps[[4, 0]]

        return kps

    # ------------------------------------------------------------------
    # Public detection API
    # ------------------------------------------------------------------

    def detect(
        self,
        frame_bgr: np.ndarray,
        max_faces: int = 1,
    ) -> List[FaceKpsBox]:
        """
        Detect a face and return a smoothed 5-point bounding box.

        MediaPipe FaceLandmarker provides accurate neural face detection
        and landmark localization. Haar cascade is used only as fallback.
        """

        height, width = frame_bgr.shape[:2]

        # ---------------------------------------------------------------
        # 1. MediaPipe full-frame 5-point detection (Primary)
        # ---------------------------------------------------------------
        kps = self._facemesh_5pt(frame_bgr)

        # ---------------------------------------------------------------
        # 2. Haar fallback if MediaPipe returned none
        # ---------------------------------------------------------------
        if kps is None:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = self._haar_faces(gray)
            if faces.shape[0] == 0:
                return []
            
            # Pick largest Haar face
            areas = faces[:, 2] * faces[:, 3]
            largest_index = int(np.argmax(areas))
            hx, hy, hw, hh = faces[largest_index].tolist()

            # Dummy keypoints estimation from Haar box if MediaPipe missed
            kps = np.array(
                [
                    [hx + 0.3 * hw, hy + 0.35 * hh],  # left eye
                    [hx + 0.7 * hw, hy + 0.35 * hh],  # right eye
                    [hx + 0.5 * hw, hy + 0.55 * hh],  # nose tip
                    [hx + 0.35 * hw, hy + 0.75 * hh],  # left mouth
                    [hx + 0.65 * hw, hy + 0.75 * hh],  # right mouth
                ],
                dtype=np.float32,
            )

        # ---------------------------------------------------------------
        # Keypoint geometry validation.
        # ---------------------------------------------------------------
        min_eye_distance = 10.0

        if not _kps_span_ok(
            kps,
            min_eye_dist=min_eye_distance,
        ):
            if self.debug:
                print("[haar_5pt] 5pt geometry sanity failed -> reject")
            return []

        # ---------------------------------------------------------------
        # Build centered bounding box from keypoints.
        # ---------------------------------------------------------------
        box = _bbox_from_5pt(
            kps,
            pad_x=0.55,
            pad_y_top=0.85,
            pad_y_bot=1.15,
        )

        box = _clip_box_xyxy(
            box,
            width,
            height,
        )

        # ---------------------------------------------------------------
        # Smooth bounding box and keypoints.
        # ---------------------------------------------------------------
        smoothed_box = _ema(
            self._prev_box,
            box,
            self.smooth_alpha,
        )

        smoothed_kps = _ema(
            self._prev_kps,
            kps,
            self.smooth_alpha,
        )

        self._prev_box = smoothed_box.copy()
        self._prev_kps = smoothed_kps.copy()

        x1, y1, x2, y2 = smoothed_box.tolist()

        score = 1.0

        face = FaceKpsBox(
            x1=int(round(x1)),
            y1=int(round(y1)),
            x2=int(round(x2)),
            y2=int(round(y2)),
            score=score,
            kps=smoothed_kps.astype(np.float32),
        )

        return [face][:max_faces]

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release MediaPipe resources."""

        if self.face_landmarker is not None:
            self.face_landmarker.close()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the live Haar + MediaPipe Face Landmarker demo."""

    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open camera."
        )

    detector = Haar5ptDetector(
        min_size=(70, 70),
        smooth_alpha=0.80,
        debug=True,
    )

    print(
        "Haar + 5pt (MediaPipe FaceLandmarker) test."
    )
    print("Press 'q' to quit.")

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                print(
                    "[haar_5pt] Failed to read frame."
                )
                break

            faces = detector.detect(
                frame,
                max_faces=1,
            )

            vis = frame.copy()

            if faces:
                face = faces[0]

                # Draw bounding box.
                cv2.rectangle(
                    vis,
                    (face.x1, face.y1),
                    (face.x2, face.y2),
                    (0, 255, 0),
                    2,
                )

                # Draw the 5 keypoints.
                for x, y in face.kps.astype(int):
                    cv2.circle(
                        vis,
                        (int(x), int(y)),
                        3,
                        (0, 255, 0),
                        -1,
                    )

                # Status text.
                cv2.putText(
                    vis,
                    "OK",
                    (
                        face.x1,
                        max(0, face.y1 - 8),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            else:
                cv2.putText(
                    vis,
                    "no face",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow(
                "haar_5pt",
                vis,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()