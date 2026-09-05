"""Clothing / attire classification service.

Detects traditional Arabic dress versus modern clothing using lightweight
computer-vision heuristics:

- Arabic male (thobe / kandura): long, mostly white/light/cream garment.
- Arabic female (abaya): long, mostly black garment.
- Modern dress: colorful, varied, shorter or fitted silhouettes.

The classifier also returns human-readable attributes (coverage, sleeve length,
fit, formality, dominant colour) so the UI can show a richer description.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Attire category constants.
ATTIRE_ARABIC_MALE = 'arabic_male'
ATTIRE_ARABIC_FEMALE = 'arabic_female'
ATTIRE_MODERN = 'modern_dress'
ATTIRE_UNKNOWN = 'unknown'

ATTIRE_CHOICES = [
    (ATTIRE_UNKNOWN, 'Unknown'),
    (ATTIRE_ARABIC_MALE, 'Arabic dress (Thobe/Kandura)'),
    (ATTIRE_ARABIC_FEMALE, 'Arabic dress (Abaya)'),
    (ATTIRE_MODERN, 'Modern dress'),
]

_ATTIRE_LABELS = dict(ATTIRE_CHOICES)


class ClothingService:
    """Singleton-like service for attribute-based attire classification."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def classify(self, frame, x1, y1, x2, y2, gender='unknown'):
        """Classify the clothing inside a person bounding box.

        Returns a dict with:
            - category: one of the ATTIRE_* constants
            - label: human-readable label
            - attributes: dict of clothing attributes
        """
        try:
            return self._classify(frame, x1, y1, x2, y2, gender)
        except Exception as exc:
            logger.warning('Attire classification failed: %s', exc)
            return self._unknown_result()

    def _classify(self, frame, x1, y1, x2, y2, gender):
        h, w = frame.shape[:2]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 <= x1 or y2 <= y1:
            return self._unknown_result()

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return self._unknown_result()

        crop = cv2.GaussianBlur(crop, (5, 5), 0)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)

        # Color masks in HSV.
        white_mask = cv2.inRange(hsv, np.array([0, 0, 170]), np.array([180, 55, 255]))
        cream_mask = cv2.inRange(hsv, np.array([15, 15, 140]), np.array([45, 130, 255]))
        black_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 45]))
        dark_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 75]))

        total = crop.shape[0] * crop.shape[1]
        white_ratio = cv2.countNonZero(white_mask) / total
        cream_ratio = cv2.countNonZero(cream_mask) / total
        black_ratio = cv2.countNonZero(black_mask) / total
        dark_ratio = cv2.countNonZero(dark_mask) / total

        # Estimate coverage / silhouette.
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, fg_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

        fg_pixels = max(1, cv2.countNonZero(fg_mask))
        fg_y, fg_x = np.where(fg_mask > 0)

        coverage = 'unknown'
        fit = 'unknown'
        if fg_y.size > 10 and fg_x.size > 10:
            y_span = int(fg_y.max() - fg_y.min())
            x_span = int(fg_x.max() - fg_x.min())
            box_h = y2 - y1
            box_w = x2 - x1

            if box_h > 0:
                vertical_coverage = y_span / box_h
                if vertical_coverage > 0.82:
                    coverage = 'full_length'
                elif vertical_coverage > 0.55:
                    coverage = 'knee_length'
                else:
                    coverage = 'short'

            if x_span > 0 and box_h > 0:
                aspect = box_h / max(1, box_w)
                # Loose garments have a larger horizontal-to-vertical silhouette.
                if aspect < 1.9 or (x_span / max(1, box_w)) > 0.75:
                    fit = 'loose'
                else:
                    fit = 'fitted'

        # Dominant colour using K-means on the foreground.
        dominant_colour = self._dominant_colour(crop, fg_mask)

        # Texture / colourfulness: low variance -> plain/formal; high -> casual/patterned.
        l_channel = lab[:, :, 0]
        fg_l = l_channel[fg_mask > 0]
        colour_variance = float(np.std(fg_l)) if fg_l.size > 0 else 0.0
        if colour_variance < 18:
            formality = 'formal_traditional'
        elif colour_variance < 35:
            formality = 'business_casual'
        else:
            formality = 'casual'

        # Sleeve length heuristics based on upper-body coverage.
        sleeve_length = self._estimate_sleeve_length(crop, fg_mask)

        # Decide category.
        light_ratio = white_ratio + cream_ratio
        box_aspect = (y2 - y1) / max(1, x2 - x1)
        category = ATTIRE_UNKNOWN
        confidence = 0.0

        # Strong colour-only cues for traditional Arabic dress.  Even if
        # foreground extraction fails (e.g. uniform-colour test images or
        # very plain garments), a tall, mostly-white or mostly-black box is
        # very likely a thobe or abaya.
        is_tall = box_aspect >= 1.3
        is_loose_or_unknown = fit in ('loose', 'unknown')

        if is_tall and light_ratio > 0.50 and gender in ('male', 'unknown'):
            category = ATTIRE_ARABIC_MALE
            confidence = min(0.98, light_ratio + 0.25)
        elif is_tall and dark_ratio > 0.50 and gender in ('female', 'unknown'):
            category = ATTIRE_ARABIC_FEMALE
            confidence = min(0.98, dark_ratio + 0.20)

        # Reinforce with coverage / fit when those signals are available.
        if category == ATTIRE_ARABIC_MALE and coverage not in ('full_length', 'knee_length'):
            if coverage != 'unknown':
                category = ATTIRE_MODERN
                confidence = max(0.45, light_ratio)
        if category == ATTIRE_ARABIC_FEMALE and coverage not in ('full_length', 'knee_length'):
            if coverage != 'unknown':
                category = ATTIRE_MODERN
                confidence = max(0.45, dark_ratio)

        # Disambiguation: if gender is known and strongly conflicts with the colour cue,
        # prefer modern dress when the colour cue is weak.
        if category != ATTIRE_UNKNOWN:
            if gender == 'female' and category == ATTIRE_ARABIC_MALE and light_ratio < 0.65:
                category = ATTIRE_MODERN
            elif gender == 'male' and category == ATTIRE_ARABIC_FEMALE and black_ratio < 0.60:
                category = ATTIRE_MODERN

        if category == ATTIRE_UNKNOWN:
            category = ATTIRE_MODERN
            confidence = max(0.45, 1.0 - max(light_ratio, dark_ratio))

        return {
            'category': category,
            'label': _ATTIRE_LABELS.get(category, category),
            'confidence': round(confidence, 2),
            'attributes': {
                'dominant_colour': dominant_colour,
                'coverage': coverage,
                'sleeve_length': sleeve_length,
                'fit': fit,
                'formality': formality,
                'colour_variance': round(colour_variance, 2),
                'white_ratio': round(white_ratio, 2),
                'black_ratio': round(black_ratio, 2),
            },
        }

    @staticmethod
    def _dominant_colour(crop, fg_mask):
        """Return a human-readable dominant colour name for the foreground."""
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        fg_pixels = hsv[fg_mask > 0]
        if fg_pixels.size == 0:
            # Fallback to the whole crop when foreground segmentation fails.
            fg_pixels = hsv.reshape(-1, 3)
        if fg_pixels.size == 0:
            return 'unknown'

        # HSV averages.
        h_mean = float(np.mean(fg_pixels[:, 0]))
        s_mean = float(np.mean(fg_pixels[:, 1]))
        v_mean = float(np.mean(fg_pixels[:, 2]))

        if v_mean > 180 and s_mean < 50:
            return 'white'
        if v_mean < 50:
            return 'black'
        if s_mean < 45:
            return 'grey'
        if (15 <= h_mean <= 35) and v_mean > 100:
            return 'beige/brown'
        if h_mean <= 10 or h_mean >= 165:
            return 'red'
        if 35 <= h_mean <= 75:
            return 'green'
        if 75 <= h_mean <= 100:
            return 'blue'
        if 100 <= h_mean <= 135:
            return 'blue/purple'
        if 135 <= h_mean <= 165:
            return 'purple/pink'
        return 'multicolour'

    @staticmethod
    def _estimate_sleeve_length(crop, fg_mask):
        """Rough sleeve-length estimate from the upper half of the crop."""
        h, w = crop.shape[:2]
        if h < 20 or w < 20:
            return 'unknown'

        upper = fg_mask[: h // 2, :]
        cols = np.count_nonzero(upper, axis=0)
        if cols.size == 0 or np.max(cols) == 0:
            return 'unknown'

        # Look at left/right edges of the upper body.
        left_edge = np.mean(cols[: max(1, w // 4)]) / max(1, h // 2)
        right_edge = np.mean(cols[-max(1, w // 4):]) / max(1, h // 2)
        edge_density = (left_edge + right_edge) / 2

        if edge_density > 0.55:
            return 'long_sleeve'
        if edge_density > 0.25:
            return 'short_sleeve'
        return 'sleeveless'

    @staticmethod
    def _unknown_result():
        return {
            'category': ATTIRE_UNKNOWN,
            'label': _ATTIRE_LABELS[ATTIRE_UNKNOWN],
            'confidence': 0.0,
            'attributes': {
                'dominant_colour': 'unknown',
                'coverage': 'unknown',
                'sleeve_length': 'unknown',
                'fit': 'unknown',
                'formality': 'unknown',
                'colour_variance': 0.0,
                'white_ratio': 0.0,
                'black_ratio': 0.0,
            },
        }


# Module-level singleton accessor.
_clothing_service = None


def get_clothing_service():
    global _clothing_service
    if _clothing_service is None:
        _clothing_service = ClothingService()
    return _clothing_service
