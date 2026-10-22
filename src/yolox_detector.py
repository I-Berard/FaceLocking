# src/yolox_detector.py
"""
YOLOX ONNX Body Detector optimized for CPU execution.

Features:
    - Letterbox pre-processing preserving aspect ratio with 114 padding.
    - Vectorized grid decoding and NMS post-processing for COCO class 0 (person).
    - CPUExecutionProvider with multi-threaded intra-op execution.
    - AsyncYOLOXBodyDetector for non-blocking background frame inference.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None


YOLOX_NANO_DOWNLOAD_URL = (
    "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx"
)
YOLOX_TINY_DOWNLOAD_URL = (
    "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx"
)


def preprocess_yolox(
    img: np.ndarray,
    input_size: Tuple[int, int] = (416, 416),
    swap: Tuple[int, int, int] = (2, 0, 1),
) -> Tuple[np.ndarray, float]:
    """
    Letterbox pad and resize image to fit input_size while maintaining aspect ratio.

    Returns:
        (chw_float_image, scale_ratio)
    """
    h, w = img.shape[:2]
    target_h, target_w = input_size

    padded_img = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    r = min(target_h / h, target_w / w)
    new_w = int(w * r)
    new_h = int(h * r)

    resized_img = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR,
    )
    padded_img[:new_h, :new_w] = resized_img

    image = padded_img.astype(np.float32)
    image = np.transpose(image, swap)  # HWC -> CHW
    return image, r


def postprocess_yolox(
    outputs: np.ndarray,
    input_size: Tuple[int, int],
    ratio: float,
    orig_shape: Tuple[int, int],
    conf_thresh: float = 0.35,
    nms_thresh: float = 0.45,
    target_class_id: int = 0,  # 0 is 'person' in COCO
) -> List[Tuple[int, int, int, int]]:
    """
    Decode raw YOLOX grid outputs into XYXY bounding boxes for the target class.

    Args:
        outputs: Raw model output tensor of shape (1, num_anchors, 85).
        input_size: (H, W) of model input.
        ratio: Scale ratio from preprocessing.
        orig_shape: (H, W) of original input image.
        conf_thresh: Minimum score threshold for detection.
        nms_thresh: Non-maximum suppression IoU threshold.
        target_class_id: Class ID to filter for (0 = person).

    Returns:
        List of (x1, y1, x2, y2) bounding boxes clamped to orig_shape.
    """
    predictions = np.copy(outputs[0])  # (num_anchors, 85)

    grids = []
    expanded_strides = []
    strides = [8, 16, 32]

    hsizes = [input_size[0] // s for s in strides]
    wsizes = [input_size[1] // s for s in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)[0]
    expanded_strides = np.concatenate(expanded_strides, 1)[0]

    # Decode cx, cy, w, h
    predictions[:, :2] = (predictions[:, :2] + grids) * expanded_strides
    predictions[:, 2:4] = np.exp(predictions[:, 2:4]) * expanded_strides

    # Scores: objectness * class_probability
    obj_conf = predictions[:, 4:5]
    cls_conf = predictions[:, 5:]
    scores = obj_conf * cls_conf

    # Filter for target class
    person_scores = scores[:, target_class_id]
    valid_mask = person_scores > conf_thresh
    if not np.any(valid_mask):
        return []

    valid_boxes = predictions[valid_mask, :4]
    valid_scores = person_scores[valid_mask]

    # Convert cxcywh -> xywh (for cv2.dnn.NMSBoxes)
    boxes_xywh = np.zeros_like(valid_boxes)
    boxes_xywh[:, 0] = (valid_boxes[:, 0] - valid_boxes[:, 2] / 2.0) / ratio
    boxes_xywh[:, 1] = (valid_boxes[:, 1] - valid_boxes[:, 3] / 2.0) / ratio
    boxes_xywh[:, 2] = valid_boxes[:, 2] / ratio
    boxes_xywh[:, 3] = valid_boxes[:, 3] / ratio

    indices = cv2.dnn.NMSBoxes(
        boxes_xywh.tolist(),
        valid_scores.tolist(),
        score_threshold=conf_thresh,
        nms_threshold=nms_thresh,
    )

    if len(indices) == 0:
        return []

    orig_h, orig_w = orig_shape
    results: List[Tuple[int, int, int, int]] = []
    for idx in np.array(indices).flatten():
        bx, by, bw, bh = boxes_xywh[idx]
        x1 = int(max(0, min(bx, orig_w - 1)))
        y1 = int(max(0, min(by, orig_h - 1)))
        x2 = int(max(0, min(bx + bw, orig_w - 1)))
        y2 = int(max(0, min(by + bh, orig_h - 1)))
        if x2 > x1 and y2 > y1:
            results.append((x1, y1, x2, y2))

    return results


class YOLOXBodyDetector:
    """
    Whole-person detector using YOLOX ONNX model running on CPU.
    """

    def __init__(
        self,
        model_path: str = "models/yolox_nano.onnx",
        input_size: Tuple[int, int] = (416, 416),
        conf_thresh: float = 0.35,
        nms_thresh: float = 0.45,
        intra_op_num_threads: int = 4,
    ):
        self.model_path = Path(model_path)
        self.input_size = input_size
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"\n[YOLOXBodyDetector] Model file not found at: '{self.model_path}'.\n"
                f"Please download the YOLOX ONNX model and place it in the models directory:\n"
                f"  Download URL: {YOLOX_NANO_DOWNLOAD_URL}\n"
                f"  Target path:  {self.model_path.resolve()}\n"
                f"  Example command: curl -L -o {self.model_path} {YOLOX_NANO_DOWNLOAD_URL}\n"
            )

        if ort is None:
            raise ImportError("onnxruntime is required for YOLOXBodyDetector. Please install onnxruntime.")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = intra_op_num_threads
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect persons in frame and return XYXY bounding boxes.
        """
        orig_shape = frame.shape[:2]
        img_blob, ratio = preprocess_yolox(frame, self.input_size)
        img_blob = np.ascontiguousarray(img_blob[None, :, :, :], dtype=np.float32)

        outputs = self.session.run([self.output_name], {self.input_name: img_blob})
        return postprocess_yolox(
            outputs[0],
            self.input_size,
            ratio,
            orig_shape,
            conf_thresh=self.conf_thresh,
            nms_thresh=self.nms_thresh,
            target_class_id=0,
        )


class AsyncYOLOXBodyDetector:
    """
    Asynchronous background wrapper for YOLOXBodyDetector.

    Runs the YOLOX inference in a background thread so the camera capture and
    UI display loop never blocks.
    """

    def __init__(self, detector: Optional[YOLOXBodyDetector] = None, **kwargs):
        self.detector = detector if detector is not None else YOLOXBodyDetector(**kwargs)
        self.is_async = True
        self._lock = threading.Lock()
        self._latest_bodies: List[Tuple[int, int, int, int]] = []
        self._pending_frame: Optional[np.ndarray] = None
        self._running = True
        self._has_new_frame = threading.Event()
        self._thread = threading.Thread(
            target=self._worker, daemon=True, name="AsyncYOLOXWorker"
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
