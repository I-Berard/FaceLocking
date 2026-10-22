# tests/test_yolox_detector.py
"""
Unit tests for YOLOX ONNX detector pre-processing, post-processing, and error handling.
"""

import unittest
import numpy as np

from src.yolox_detector import (
    YOLOXBodyDetector,
    postprocess_yolox,
    preprocess_yolox,
)


class TestYOLOXDetector(unittest.TestCase):
    """Test YOLOX preprocessing, postprocessing, and missing model detection."""

    def test_preprocess_letterbox(self):
        """Preprocess should letterbox pad and return CHW float32 image."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        blob, ratio = preprocess_yolox(img, input_size=(416, 416))

        self.assertEqual(blob.shape, (3, 416, 416))
        self.assertEqual(blob.dtype, np.float32)
        # Ratio for 640 -> 416 should be 416 / 640 = 0.65
        self.assertAlmostEqual(ratio, 416.0 / 640.0, places=4)

    def test_postprocess_grid_decoding(self):
        """Synthetic raw outputs should decode into valid bounding boxes."""
        input_size = (416, 416)
        num_anchors = (416 // 8) ** 2 + (416 // 16) ** 2 + (416 // 32) ** 2  # 3549
        raw_output = np.zeros((1, num_anchors, 85), dtype=np.float32)

        # Place a high-confidence person detection at anchor 0
        raw_output[0, 0, 0] = 0.0   # dx (relative to grid)
        raw_output[0, 0, 1] = 0.0   # dy
        raw_output[0, 0, 2] = 2.0   # log(w/stride) -> exp(2.0)*8 ~= 59.1
        raw_output[0, 0, 3] = 3.0   # log(h/stride) -> exp(3.0)*8 ~= 160.6
        raw_output[0, 0, 4] = 0.95  # obj_conf
        raw_output[0, 0, 5] = 0.99  # class 0 (person) prob

        ratio = 0.5
        orig_shape = (480, 640)
        boxes = postprocess_yolox(
            raw_output,
            input_size=input_size,
            ratio=ratio,
            orig_shape=orig_shape,
            conf_thresh=0.35,
            nms_thresh=0.45,
            target_class_id=0,
        )

        self.assertGreater(len(boxes), 0)
        x1, y1, x2, y2 = boxes[0]
        self.assertGreaterEqual(x1, 0)
        self.assertGreaterEqual(y1, 0)
        self.assertLessEqual(x2, 640)
        self.assertLessEqual(y2, 480)
        self.assertGreater(x2, x1)
        self.assertGreater(y2, y1)

    def test_missing_model_raises_clear_error(self):
        """Initializing with a nonexistent path should raise FileNotFoundError with instructions."""
        with self.assertRaises(FileNotFoundError) as ctx:
            YOLOXBodyDetector(model_path="models/nonexistent_yolox.onnx")

        msg = str(ctx.exception)
        self.assertIn("download", msg.lower())
        self.assertIn("yolox", msg.lower())


if __name__ == "__main__":
    unittest.main()
