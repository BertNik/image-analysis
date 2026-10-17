from __future__ import annotations

from typing import Any

import cv2
import numpy as np

WARNING_END_TEXT = (
    "MAY CAUSE HEALTH PROBLEMS."
)

def crop_box(
    image: np.ndarray,
    box: list[int] | tuple[int, ...],
    *,
    padding: int = 4,
) -> np.ndarray | None:
    if len(box) != 4:
        return None

    try:
        x1, y1, x2, y2 = (
            int(value)
            for value in box
        )
    except (TypeError, ValueError):
        return None

    image_height, image_width = (
        image.shape[:2]
    )

    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(
        image_width,
        x2 + padding,
    )
    y2 = min(
        image_height,
        y2 + padding,
    )

    if x2 <= x1 or y2 <= y1:
        return None

    return image[
        y1:y2,
        x1:x2,
    ]


def build_text_mask(
    crop: np.ndarray,
) -> np.ndarray:
    if crop.ndim == 3:
        gray = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2GRAY,
        )
    else:
        gray = crop

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0,
    )

    _, mask = cv2.threshold(
        gray,
        0,
        255,
        (
            cv2.THRESH_BINARY_INV
            | cv2.THRESH_OTSU
        ),
    )

    return mask


def normalized_stroke_width(
    mask: np.ndarray,
) -> float | None:
    binary = (
        mask > 0
    ).astype(np.uint8)

    if not np.any(binary):
        return None

    distance = cv2.distanceTransform(
        binary,
        cv2.DIST_L2,
        5,
    )

    values = distance[
        distance > 0
    ]

    if not len(values):
        return None

    active_rows = np.flatnonzero(
        binary.any(axis=1)
    )

    if not len(active_rows):
        return None

    text_height = (
        int(active_rows[-1])
        - int(active_rows[0])
        + 1
    )

    if text_height <= 0:
        return None

    estimated_stroke_width = (
        float(np.median(values))
        * 2.0
    )

    return (
        estimated_stroke_width
        / float(text_height)
    )




def get_box(
    observation: dict[str, Any],
) -> tuple[int, int, int, int] | None:
    box = observation.get("box")

    if (
        not isinstance(box, (list, tuple))
        or len(box) != 4
    ):
        return None

    try:
        return tuple(
            int(value)
            for value in box
        )
    except (TypeError, ValueError):
        return None


def belongs_to_warning_column(
    observation: dict[str, Any],
    heading_box: tuple[int, int, int, int],
    *,
    left_tolerance: int = 50,
) -> bool:
    box = get_box(observation)

    if box is None:
        return False

    heading_left, _, _, _ = heading_box
    observation_left, _, _, _ = box

    return (
        observation_left
        >= heading_left - left_tolerance
    )

def select_warning_body_references(
    observations: list[dict[str, Any]],
    warning_line_indexes: list[int],
) -> list[dict[str, Any]]:
    if len(warning_line_indexes) < 2:
        return []

    # First warning line is the heading.
    body_indexes = set(
        warning_line_indexes[1:]
    )

    return [
        observation
        for observation in observations
        if observation.get("index")
        in body_indexes
    ]

def analyze_heading_boldness(
    *,
    image: np.ndarray,
    heading_box: list[int],
    observations: list[dict[str, Any]],
    warning_line_indexes: list[int],
    bold_ratio: float = 1.25,
    normal_ratio: float = 1.10,
) -> dict[str, Any]:
    heading_crop = crop_box(
        image,
        heading_box,
    )

    if heading_crop is None:
        return {
            "value": None,
            "reason":
                "heading_crop_unavailable",
            "method":
                "relative_stroke_width",
        }

    heading_score = (
        normalized_stroke_width(
            build_text_mask(
                heading_crop
            )
        )
    )

    if heading_score is None:
        return {
            "value": None,
            "heading_score": None,
            "reason":
                "heading_stroke_width_unavailable",
            "method":
                "relative_stroke_width",
        }

    references = (
        select_warning_body_references(
            observations,
            warning_line_indexes,
        )
    )

    reference_scores = []
    reference_evidence = []

    for observation in references:
        box = observation.get(
            "box"
        )

        if box is None:
            continue

        crop = crop_box(
            image,
            box,
        )

        if crop is None:
            continue

        score = normalized_stroke_width(
            build_text_mask(
                crop
            )
        )

        if score is None:
            continue

        reference_scores.append(
            score
        )

        reference_evidence.append(
            {
                "index":
                    observation.get(
                        "index"
                    ),

                "text":
                    observation.get(
                        "text"
                    ),

                "score":
                    score,

                "box":
                    box,
            }
        )

    if not reference_scores:
        return {
            "value": None,

            "normalized_stroke_width":
                heading_score,

            "heading_score":
                heading_score,

            "reference_score":
                None,

            "ratio":
                None,

            "reference_count":
                0,

            "reference_evidence":
                [],

            "reason":
                "no_warning_body_reference",

            "method":
                "relative_warning_stroke_width",
        }

    reference_score = float(
        np.median(
            reference_scores
        )
    )

    if reference_score <= 0:
        return {
            "value": None,
            "heading_score":
                heading_score,
            "reference_score":
                reference_score,
            "ratio":
                None,
            "reference_count":
                len(reference_scores),
            "reference_evidence":
                reference_evidence,
            "reason":
                "invalid_reference_score",
            "method":
                "relative_warning_stroke_width",
        }

    ratio = (
        heading_score
        / reference_score
    )

    if ratio >= bold_ratio:
        value = True
        reason = None

    elif ratio <= normal_ratio:
        value = False
        reason = None

    else:
        value = None
        reason = "boldness_ambiguous"

    return {
        "value":
            value,

        "normalized_stroke_width":
            heading_score,

        "heading_score":
            heading_score,

        "reference_score":
            reference_score,

        "ratio":
            ratio,

        "reference_count":
            len(reference_scores),

        "reference_evidence":
            reference_evidence,

        "reason":
            reason,

        "method":
            "relative_warning_stroke_width",
    }
