# tests/test_body_features.py
"""
Unit tests for body feature extraction, profile learning, and re-identification.
"""

import unittest
import numpy as np

from src.body_features import (
    BodyFeatureExtractor,
    BodyFeatureVector,
    BodyProfile,
    compare_feature_vectors,
)


def _make_person_image(
    top_color_bgr=(255, 0, 0),      # Default Blue shirt
    bottom_color_bgr=(0, 0, 0),     # Default Black pants
    h=200,
    w=80,
    noise=0,
) -> np.ndarray:
    """Generate a synthetic test person image with distinct top and bottom clothes."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Head/neck (0 to 15%)
    img[: int(h * 0.15), :] = (180, 190, 210)  # skin tone
    # Top/Torso (15% to 55%)
    img[int(h * 0.15): int(h * 0.55), :] = top_color_bgr
    # Bottom/Pants (55% to 92%)
    img[int(h * 0.55): int(h * 0.92), :] = bottom_color_bgr
    # Shoes (92% to 100%)
    img[int(h * 0.92):, :] = (40, 40, 40)

    if noise > 0:
        rng = np.random.default_rng(42)
        n = rng.integers(-noise, noise + 1, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + n, 0, 255).astype(np.uint8)

    return img


class TestBodyFeatures(unittest.TestCase):
    """Test feature extraction and comparison."""

    def setUp(self):
        self.extractor = BodyFeatureExtractor()

    def test_feature_extraction_structure(self):
        """Verify extracted feature vector contains all expected components."""
        img = _make_person_image()
        feat = self.extractor.extract(img, (0, 0, 80, 200))
        self.assertIsNotNone(feat)
        self.assertEqual(feat.upper.moments.shape, (6,))
        self.assertEqual(feat.upper.cdfs.shape, (3, 16))
        self.assertEqual(feat.lower.moments.shape, (6,))
        self.assertEqual(feat.lower.cdfs.shape, (3, 16))
        self.assertAlmostEqual(feat.aspect_ratio, 80 / 200.0, places=3)

    def test_same_person_high_similarity(self):
        """Same person with slight noise should have very high similarity (> 0.75)."""
        img1 = _make_person_image(top_color_bgr=(220, 30, 20), bottom_color_bgr=(30, 30, 30))
        img2 = _make_person_image(top_color_bgr=(210, 35, 25), bottom_color_bgr=(35, 30, 35), noise=5)

        feat1 = self.extractor.extract(img1, (0, 0, 80, 200))
        feat2 = self.extractor.extract(img2, (0, 0, 80, 200))

        sim = compare_feature_vectors(feat1, feat2)
        self.assertGreater(sim, 0.75, f"Expected high similarity for same person, got {sim}")

    def test_stranger_different_clothes_rejected(self):
        """A stranger wearing completely different colors should have low similarity (< 0.45)."""
        # Target: Red shirt, dark pants
        target_img = _make_person_image(top_color_bgr=(0, 0, 220), bottom_color_bgr=(20, 20, 20))
        # Stranger: Yellow/Green shirt, white pants
        stranger_img = _make_person_image(top_color_bgr=(0, 220, 220), bottom_color_bgr=(240, 240, 240))

        feat_target = self.extractor.extract(target_img, (0, 0, 80, 200))
        feat_stranger = self.extractor.extract(stranger_img, (0, 0, 80, 200))

        sim = compare_feature_vectors(feat_target, feat_stranger)
        self.assertLess(sim, 0.45, f"Expected low similarity for stranger, got {sim}")

    def test_rotation_180_invariance(self):
        """
        When turning 180 degrees, back view shares the same clothing colors and proportions.
        Similarity between front and back views must be high (> 0.80).
        """
        # Front: Blue sweater, khaki pants
        front_view = _make_person_image(top_color_bgr=(180, 100, 20), bottom_color_bgr=(120, 160, 200))
        # Back: Same sweater, same pants, back of head (hair instead of skin)
        back_view = _make_person_image(top_color_bgr=(175, 95, 25), bottom_color_bgr=(118, 158, 195))
        # Hair at the back of head instead of skin tone
        back_view[: int(200 * 0.15), :] = (20, 20, 20)

        feat_front = self.extractor.extract(front_view, (0, 0, 80, 200))
        feat_back = self.extractor.extract(back_view, (0, 0, 80, 200))

        sim = compare_feature_vectors(feat_front, feat_back)
        self.assertGreater(sim, 0.80, f"Expected high similarity for back view, got {sim}")

    def test_body_profile_accumulation_and_ready(self):
        """BodyProfile should accumulate samples and set is_ready when min_samples reached."""
        profile = BodyProfile(target_name="Alice", min_samples=5)
        self.assertFalse(profile.is_ready)
        self.assertEqual(profile.learning_progress, 0.0)

        img = _make_person_image(top_color_bgr=(200, 50, 50), bottom_color_bgr=(40, 40, 40))
        feat = self.extractor.extract(img, (0, 0, 80, 200))

        for i in range(5):
            profile.update(feat)
            self.assertEqual(profile.sample_count, i + 1)

        self.assertTrue(profile.is_ready)
        self.assertEqual(profile.learning_progress, 1.0)
        self.assertIsNotNone(profile.prototype)

        # Matching against profile
        score = profile.match(feat)
        self.assertGreater(score, 0.95)


if __name__ == "__main__":
    unittest.main()
