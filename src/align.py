# src/align.py
"""
Face alignment demo using the working pipeline:

- Haar face detection
- MediaPipe FaceMesh -> 5 keypoints
- ArcFace-style 5-point alignment -> 112x112 (or any configured size)

Run:
    python -m src.align

Keys:
    q - quit
    s - save the current aligned face to data/debug_aligned/
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from .haar_5pt import Haar5ptDetector, align_face_5pt


def _put_text(
    img: np.ndarray,
    text: str,
    xy: tuple[int, int] = (10, 30),
    scale: float = 0.8,
    thickness: int = 2,
) -> None:
    """Draw white text on an image."""
    cv2.putText(
        img,
        text,
        xy,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def _safe_imshow(
    win: str,
    img: np.ndarray | None,
) -> None:
    """Show an image if it is valid."""
    if img is not None and img.size:
        cv2.imshow(win, img)


def main(
    cam_index: int = 1,
    out_size: Tuple[int, int] = (112, 112),
    mirror: bool = True,
) -> None:
    """Run the live face alignment demo."""

    # Camera: try requested index, then fallbacks
    cap = None
    for idx in [cam_index, 0, 1, 2]:
        temp_cap = cv2.VideoCapture(idx)
        if temp_cap.isOpened():
            cap = temp_cap
            break
        temp_cap.release()

    if cap is None or not cap.isOpened():
        raise RuntimeError(f"Could not open camera {cam_index} or fallbacks 0, 1, 2")

    # Face detector (debug=False for speed)
    detector = Haar5ptDetector(
        min_size=(70, 70),
        smooth_alpha=0.80,
        debug=False,
    )

    out_w, out_h = map(int, out_size)

    # Blank aligned-face image
    blank = np.zeros(
        (out_h, out_w, 3),
        dtype=np.uint8,
    )

    last_aligned = blank.copy()

    # Output directory
    save_dir = Path("data/debug_aligned")
    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # FPS tracking
    fps_start = time.time()
    fps_frames = 0
    fps = 0.0

    print("align running.")
    print("Press 'q' to quit.")
    print("Press 's' to save the aligned face.")

    try:
        while True:
            # Capture frame
            ok, frame = cap.read()

            if not ok:
                print("[align] failed to read frame.")
                break

            if mirror:
                frame = cv2.flip(frame, 1)

            # Detect faces
            faces = detector.detect(
                frame,
                max_faces=1,
            )

            vis = frame.copy()

            if faces:
                face = faces[0]

                # Draw bounding box
                cv2.rectangle(
                    vis,
                    (face.x1, face.y1),
                    (face.x2, face.y2),
                    (0, 255, 0),
                    2,
                )

                # Draw 5 facial keypoints
                for x, y in face.kps.astype(int):
                    cv2.circle(
                        vis,
                        (int(x), int(y)),
                        3,
                        (0, 255, 0),
                        -1,
                    )

                # Align face using the 5 keypoints
                aligned, _matrix = align_face_5pt(
                    frame,
                    face.kps,
                    out_size=out_size,
                )

                # Keep the last valid aligned face
                if aligned is not None and aligned.size:
                    last_aligned = aligned

                _put_text(
                    vis,
                    "OK (Haar + FaceMesh 5pt)",
                    (10, 30),
                    0.75,
                    2,
                )

            else:
                _put_text(
                    vis,
                    "no face",
                    (10, 30),
                    0.9,
                    2,
                )

            # FPS
            fps_frames += 1

            elapsed = time.time() - fps_start

            if elapsed >= 1.0:
                fps = fps_frames / elapsed
                fps_frames = 0
                fps_start = time.time()

            _put_text(
                vis,
                f"FPS: {fps:.1f}",
                (10, 60),
                0.75,
                2,
            )

            _put_text(
                vis,
                f"warp: 5pt -> {out_w}x{out_h}",
                (10, 90),
                0.75,
                2,
            )

            # Display
            _safe_imshow(
                "align - camera",
                vis,
            )

            _safe_imshow(
                "align - aligned",
                last_aligned,
            )

            # Keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("s"):
                timestamp = int(time.time() * 1000)
                output_path = save_dir / f"{timestamp}.jpg"

                success = cv2.imwrite(
                    str(output_path),
                    last_aligned,
                )

                if success:
                    print(f"[align] saved: {output_path}")
                else:
                    print(
                        f"[align] failed to save: {output_path}"
                    )

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()