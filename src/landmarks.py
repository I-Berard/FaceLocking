# src/landmarks.py
"""
Minimal pipeline:
camera -> Haar face box -> MediaPipe FaceLandmarker -> extract 5 keypoints -> draw

Run:
    python -m src.landmarks

Keys:
    q : quit
"""

import cv2
import numpy as np
import mediapipe as mp


# MediaPipe Face Landmarker indices
IDX_LEFT_EYE = 33
IDX_RIGHT_EYE = 263
IDX_NOSE_TIP = 1
IDX_MOUTH_LEFT = 61
IDX_MOUTH_RIGHT = 291


def main():
    # --------------------------------------------------
    # Haar face detector
    # --------------------------------------------------
    cascade_path = "models/haarcascade_frontalface_default.xml"

    face = cv2.CascadeClassifier(cascade_path)

    if face.empty():
        raise RuntimeError(
            f"Failed to load cascade: {cascade_path}"
        )

    # --------------------------------------------------
    # MediaPipe Face Landmarker
    # --------------------------------------------------
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path="models/face_landmarker.task"
        ),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    landmarker = FaceLandmarker.create_from_options(options)

    # --------------------------------------------------
    # Camera
    # --------------------------------------------------
    cap = None
    for cam_idx in [1, 0, 2]:
        temp_cap = cv2.VideoCapture(cam_idx)
        if temp_cap.isOpened():
            cap = temp_cap
            break
        temp_cap.release()

    if cap is None or not cap.isOpened():
        landmarker.close()
        raise RuntimeError("Camera not opened. Tried camera indices 1, 0, 2.")

    print("MediaPipe FaceLandmarker 5pt. Press 'q' to quit.")

    timestamp_ms = 0

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                print("Failed to read frame.")
                break

            H, W = frame.shape[:2]

            # Optional Haar face detection fallback
            if not face.empty():
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(60, 60),
                )
                for (x, y, w, h) in faces:
                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + w, y + h),
                        (0, 255, 0),
                        1,
                    )

            # --------------------------------------------------
            # MediaPipe
            # --------------------------------------------------
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb,
            )

            # Timestamp must increase for every frame
            timestamp_ms += 33

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms,
            )

            # --------------------------------------------------
            # Extract 5 landmarks
            # --------------------------------------------------
            if result.face_landmarks:

                lm = result.face_landmarks[0]

                idxs = [
                    IDX_LEFT_EYE,
                    IDX_RIGHT_EYE,
                    IDX_NOSE_TIP,
                    IDX_MOUTH_LEFT,
                    IDX_MOUTH_RIGHT,
                ]

                pts = []

                for i in idxs:
                    p = lm[i]

                    pts.append([
                        p.x * W,
                        p.y * H,
                    ])

                kps = np.array(
                    pts,
                    dtype=np.float32,
                )

                # --------------------------------------------------
                # Enforce left/right ordering
                # --------------------------------------------------
                if kps[0, 0] > kps[1, 0]:
                    kps[[0, 1]] = kps[[1, 0]]

                if kps[3, 0] > kps[4, 0]:
                    kps[[3, 4]] = kps[[4, 3]]

                # --------------------------------------------------
                # Draw 5 points
                # --------------------------------------------------
                for px, py in kps.astype(int):
                    cv2.circle(
                        frame,
                        (int(px), int(py)),
                        4,
                        (0, 255, 0),
                        -1,
                    )

                cv2.putText(
                    frame,
                    "5pt",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )

            # --------------------------------------------------
            # Display
            # --------------------------------------------------
            cv2.imshow(
                "5pt Landmarks",
                frame,
            )

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    finally:
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()