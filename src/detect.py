# src/detect.py
import cv2
import numpy as np
import mediapipe as mp


def main():
    landmark_model_path = "models/face_landmarker.task"

    # Initialize MediaPipe FaceLandmarker
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=landmark_model_path),
        running_mode=RunningMode.VIDEO,
        num_faces=5,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    landmarker = FaceLandmarker.create_from_options(options)

    # Optional Haar fallback
    cascade_path = "models/haarcascade_frontalface_default.xml"
    haar_cascade = cv2.CascadeClassifier(cascade_path)

    # Try camera indices 1, 0, 2
    cap = None
    for cam_idx in [1, 0, 2]:
        temp_cap = cv2.VideoCapture(cam_idx)
        if temp_cap.isOpened():
            cap = temp_cap
            break
        temp_cap.release()

    if cap is None or not cap.isOpened():
        landmarker.close()
        raise RuntimeError("Camera not opened. Checked camera indices 1, 0, 2.")

    print("MediaPipe face detect. Press 'q' to quit.")
    timestamp_ms = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms += 33

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            detected_faces = []
            if result.face_landmarks:
                for face_lms in result.face_landmarks:
                    pts = np.array([[lm.x * w, lm.y * h] for lm in face_lms], dtype=np.float32)
                    x1, y1 = np.min(pts, axis=0)
                    x2, y2 = np.max(pts, axis=0)

                    # Add padding
                    pad_w = (x2 - x1) * 0.15
                    pad_h = (y2 - y1) * 0.15
                    x1 = int(max(0, x1 - pad_w))
                    y1 = int(max(0, y1 - pad_h))
                    x2 = int(min(w - 1, x2 + pad_w))
                    y2 = int(min(h - 1, y2 + pad_h))

                    detected_faces.append((x1, y1, x2 - x1, y2 - y1))

            # Fallback to Haar if MediaPipe returned no faces and Haar loaded
            if not detected_faces and not haar_cascade.empty():
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                haar_faces = haar_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
                )
                if haar_faces is not None and len(haar_faces) > 0:
                    detected_faces = [(x, y, w_box, h_box) for (x, y, w_box, h_box) in haar_faces]

            for (x, y, box_w, box_h) in detected_faces:
                cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), (0, 255, 0), 2)

            cv2.imshow("Face Detection", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()