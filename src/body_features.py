# src/body_features.py
"""
Whole-body appearance feature extraction and online learning.

Extracts rich multi-zone color, texture, and morphological features from
detected body bounding boxes to enable:
    1. Re-identification when a person turns 180 degrees (face hidden).
    2. Rejection of unknown persons / strangers entering facing away.
    3. Smooth temporal tracking continuity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class ZoneFeatures:
    """Appearance features for a specific anatomical zone (upper torso or lower body)."""

    moments: np.ndarray       # Normalized mean & std dev for L, a, b (6,)
    cdfs: np.ndarray          # 1D cumulative distributions for L, a, b (3, num_bins)
    texture: float            # Sobel gradient energy


@dataclass
class BodyFeatureVector:
    """Representation of a person's body appearance across zones."""

    upper: ZoneFeatures
    lower: ZoneFeatures
    aspect_ratio: float       # width / height of body bounding box


class BodyFeatureExtractor:
    """
    Extracts multi-zone appearance features from a cropped body bounding box.
    """

    def __init__(
        self,
        num_bins: int = 16,
        upper_range: Tuple[float, float] = (0.15, 0.55),
        lower_range: Tuple[float, float] = (0.55, 0.92),
    ):
        self.num_bins = num_bins
        self.upper_range = upper_range
        self.lower_range = lower_range

    def _extract_zone(self, patch_bgr: np.ndarray) -> ZoneFeatures:
        """Extract moments, channel CDFs, and texture from a BGR image patch."""
        if patch_bgr.size == 0 or patch_bgr.shape[0] < 4 or patch_bgr.shape[1] < 4:
            return ZoneFeatures(
                moments=np.zeros(6, dtype=np.float32),
                cdfs=np.zeros((3, self.num_bins), dtype=np.float32),
                texture=0.0,
            )

        lab = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Perceptual Lab moments (mean and standard deviation)
        mean_lab = np.mean(lab, axis=(0, 1)) / 255.0
        std_lab = np.std(lab, axis=(0, 1)) / 128.0
        moments = np.concatenate([mean_lab, std_lab]).astype(np.float32)

        # 2. Cumulative channel distributions (robust to bin boundary quantization)
        lab_uint8 = np.clip(lab, 0, 255).astype(np.uint8)
        cdfs = []
        for ch in range(3):
            hist = cv2.calcHist(
                [lab_uint8], [ch], None, [self.num_bins], [0, 256]
            ).flatten()
            total = float(np.sum(hist))
            if total > 0:
                cdf = np.cumsum(hist / total)
            else:
                cdf = np.zeros(self.num_bins, dtype=np.float32)
            cdfs.append(cdf)
        cdfs_arr = np.array(cdfs, dtype=np.float32)

        # 3. Texture / edge gradient energy
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        texture = float(np.mean(cv2.magnitude(gx, gy)) / 255.0)

        return ZoneFeatures(moments=moments, cdfs=cdfs_arr, texture=texture)

    def extract(
        self, frame: np.ndarray, body_box: Tuple[int, int, int, int]
    ) -> Optional[BodyFeatureVector]:
        """
        Extract features from the region of frame defined by body_box (XYXY).
        """
        frame_h, frame_w = frame.shape[:2]
        x1, y1, x2, y2 = body_box

        # Clamp to frame boundaries
        x1 = max(0, min(x1, frame_w - 1))
        y1 = max(0, min(y1, frame_h - 1))
        x2 = max(0, min(x2, frame_w))
        y2 = max(0, min(y2, frame_h))

        bw = x2 - x1
        bh = y2 - y1
        if bw < 12 or bh < 24:
            return None

        body_crop = frame[y1:y2, x1:x2]

        # Segment into vertical zones
        uy1 = int(bh * self.upper_range[0])
        uy2 = int(bh * self.upper_range[1])
        ly1 = int(bh * self.lower_range[0])
        ly2 = int(bh * self.lower_range[1])

        upper_patch = body_crop[uy1:uy2, :]
        lower_patch = body_crop[ly1:ly2, :]

        upper_feat = self._extract_zone(upper_patch)
        lower_feat = self._extract_zone(lower_patch)
        aspect_ratio = float(bw) / float(bh)

        return BodyFeatureVector(
            upper=upper_feat,
            lower=lower_feat,
            aspect_ratio=aspect_ratio,
        )


def _compare_zones(z1: ZoneFeatures, z2: ZoneFeatures) -> float:
    """Compute similarity between two zone feature sets in [0.0, 1.0]."""
    # 1. Color moments distance
    dist_m = float(np.linalg.norm(z1.moments - z2.moments))
    sim_m = math.exp(-dist_m / 0.14)

    # 2. Cumulative distribution similarity (Wasserstein / EMD)
    emd = float(np.mean(np.sum(np.abs(z1.cdfs - z2.cdfs), axis=1) / float(z1.cdfs.shape[1])))
    sim_c = max(0.0, 1.0 - 2.0 * emd)

    # 3. Texture similarity
    diff_t = abs(z1.texture - z2.texture)
    sim_t = math.exp(-diff_t / 0.15)

    return float(np.clip(0.50 * sim_m + 0.40 * sim_c + 0.10 * sim_t, 0.0, 1.0))


def compare_feature_vectors(
    cand: BodyFeatureVector,
    ref: BodyFeatureVector,
    weights: Tuple[float, float, float] = (0.45, 0.45, 0.10),
) -> float:
    """
    Compute similarity between two body feature vectors in range [0.0, 1.0].

    Weights correspond to:
        (upper_body_sim, lower_body_sim, aspect_ratio_sim)
    """
    sim_upper = _compare_zones(cand.upper, ref.upper)
    sim_lower = _compare_zones(cand.lower, ref.lower)

    diff_ar = abs(cand.aspect_ratio - ref.aspect_ratio)
    sim_aspect = math.exp(-diff_ar / 0.18)

    w_up, w_lo, w_ar = weights
    total_w = w_up + w_lo + w_ar

    composite = (w_up * sim_upper + w_lo * sim_lower + w_ar * sim_aspect) / total_w
    return float(np.clip(composite, 0.0, 1.0))


class BodyProfile:
    """
    Accumulates and maintains the learned appearance profile of a target person.
    """

    def __init__(
        self,
        target_name: str,
        min_samples: int = 10,
        max_gallery: int = 40,
        ema_alpha: float = 0.18,
    ):
        self.target_name = target_name
        self.min_samples = min_samples
        self.max_gallery = max_gallery
        self.ema_alpha = ema_alpha

        self.samples: List[BodyFeatureVector] = []
        self.prototype: Optional[BodyFeatureVector] = None
        self.sample_count: int = 0

    @property
    def is_ready(self) -> bool:
        """True when minimum required feature samples have been learned."""
        return self.sample_count >= self.min_samples

    @property
    def learning_progress(self) -> float:
        """Progress fraction from 0.0 (empty) to 1.0 (ready)."""
        if self.min_samples <= 0:
            return 1.0
        return float(min(1.0, self.sample_count / float(self.min_samples)))

    def update(self, feat: BodyFeatureVector) -> None:
        """
        Incorporate a new body feature sample into the learned profile.
        """
        self.sample_count += 1
        self.samples.append(feat)
        if len(self.samples) > self.max_gallery:
            self.samples.pop(0)

        # Update running EMA prototype
        if self.prototype is None:
            self.prototype = BodyFeatureVector(
                upper=ZoneFeatures(
                    moments=feat.upper.moments.copy(),
                    cdfs=feat.upper.cdfs.copy(),
                    texture=feat.upper.texture,
                ),
                lower=ZoneFeatures(
                    moments=feat.lower.moments.copy(),
                    cdfs=feat.lower.cdfs.copy(),
                    texture=feat.lower.texture,
                ),
                aspect_ratio=feat.aspect_ratio,
            )
        else:
            a = self.ema_alpha
            self.prototype.upper.moments = (
                (1.0 - a) * self.prototype.upper.moments + a * feat.upper.moments
            )
            self.prototype.upper.cdfs = (
                (1.0 - a) * self.prototype.upper.cdfs + a * feat.upper.cdfs
            )
            self.prototype.upper.texture = (
                (1.0 - a) * self.prototype.upper.texture + a * feat.upper.texture
            )

            self.prototype.lower.moments = (
                (1.0 - a) * self.prototype.lower.moments + a * feat.lower.moments
            )
            self.prototype.lower.cdfs = (
                (1.0 - a) * self.prototype.lower.cdfs + a * feat.lower.cdfs
            )
            self.prototype.lower.texture = (
                (1.0 - a) * self.prototype.lower.texture + a * feat.lower.texture
            )

            self.prototype.aspect_ratio = (
                (1.0 - a) * self.prototype.aspect_ratio + a * feat.aspect_ratio
            )

    def match(self, candidate_feat: BodyFeatureVector) -> float:
        """
        Match a candidate body against this profile.

        Returns maximum similarity between candidate and (EMA prototype + sample gallery).
        """
        if self.prototype is None:
            return 0.0

        # Score against running prototype
        proto_score = compare_feature_vectors(candidate_feat, self.prototype)

        # Also check score against recent sample gallery (catches diverse angles/poses)
        gallery_scores = [
            compare_feature_vectors(candidate_feat, s) for s in self.samples[-15:]
        ]
        best_gallery = max(gallery_scores) if gallery_scores else 0.0

        return max(proto_score, best_gallery)

    def reset(self) -> None:
        """Clear learned features."""
        self.samples.clear()
        self.prototype = None
        self.sample_count = 0
