import cv2
import numpy as np
import pytest

import lib.analyze_image as analyze_image_module
import lib.resolver.visual as visual

from lib.resolver.visual import (
    crop_box,
    normalized_stroke_width,
)
from lib.resolver.class_type import (
    complete_class_type_visual_block,
)

# ============================================================
# crop_box
# ============================================================


@pytest.mark.parametrize(
    "box",
    [
        [],
        [10],
        [10, 20],
        [10, 20, 30],
        [10, 20, 30, 40, 50],
    ],
)
def test_crop_box_rejects_invalid_box_length(
    box,
):
    image = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    assert crop_box(
        image,
        box,
    ) is None


@pytest.mark.parametrize(
    "box",
    [
        # x2 < x1
        [40, 10, 20, 30],

        # x2 == x1
        [20, 10, 20, 30],

        # y2 < y1
        [10, 40, 30, 20],

        # y2 == y1
        [10, 20, 30, 20],
    ],
)
def test_crop_box_returns_none_for_non_positive_area(
    box,
):
    image = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    result = crop_box(
        image,
        box,
        padding=0,
    )

    assert result is None


@pytest.mark.parametrize(
    "box",
    [
        # ValueError
        [10, 20, "bad", 40],

        # TypeError
        [10, None, 30, 40],

        # TypeError
        [object(), 20, 30, 40],
    ],
)
def test_crop_box_returns_none_for_invalid_coordinate_types(
    box,
):
    image = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    result = crop_box(
        image,
        box,
        padding=0,
    )

    assert result is None


def test_crop_box_returns_expected_region():
    image = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    image[
        20:40,
        10:30,
    ] = 255

    result = crop_box(
        image,
        [10, 20, 30, 40],
        padding=0,
    )

    assert result is not None

    assert result.shape == (
        20,
        20,
        3,
    )

    assert np.all(
        result == 255
    )


# ============================================================
# normalized_stroke_width
# ============================================================


def test_normalized_stroke_width_returns_none_for_empty_mask():
    binary = np.zeros(
        (40, 100),
        dtype=np.uint8,
    )

    result = normalized_stroke_width(
        binary
    )

    assert result is None


def test_normalized_stroke_width_returns_positive_score():
    binary = np.zeros(
        (40, 100),
        dtype=np.uint8,
    )

    cv2.putText(
        binary,
        "TEST",
        (5, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        255,
        2,
        cv2.LINE_AA,
    )

    result = normalized_stroke_width(
        binary
    )

    assert result is not None
    assert result > 0.0


def test_thicker_text_has_larger_normalized_stroke_width():
    thin = np.zeros(
        (60, 200),
        dtype=np.uint8,
    )

    thick = np.zeros(
        (60, 200),
        dtype=np.uint8,
    )

    cv2.putText(
        thin,
        "WARNING",
        (5, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        255,
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        thick,
        "WARNING",
        (5, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        255,
        3,
        cv2.LINE_AA,
    )

    thin_score = (
        normalized_stroke_width(
            thin
        )
    )

    thick_score = (
        normalized_stroke_width(
            thick
        )
    )

    assert thin_score is not None
    assert thick_score is not None

    assert thick_score > thin_score


# ============================================================
# analyze_heading_boldness
# ============================================================


def test_heading_boldness_returns_ambiguous_for_middle_ratio(
    monkeypatch,
):
    image = np.zeros(
        (200, 500, 3),
        dtype=np.uint8,
    )

    observations = [
        {
            "index": 0,
            "text":
                "GOVERNMENT WARNING:",
            "box": [
                10,
                10,
                250,
                40,
            ],
        },
        {
            "index": 1,
            "text":
                (
                    "(1) According to the "
                    "Surgeon General,"
                ),
            "box": [
                10,
                50,
                350,
                80,
            ],
        },
    ]

    # analyze_heading_boldness should obtain:
    #
    # heading_score   = 1.18
    # reference_score = 1.00
    # ratio           = 1.18
    #
    # With:
    # normal_ratio = 1.10
    # bold_ratio   = 1.25
    #
    # the result must be ambiguous.
    scores = iter(
        [
            1.18,
            1.00,
        ]
    )

    monkeypatch.setattr(
        visual,
        "normalized_stroke_width",
        lambda binary:
            next(scores),
    )

    result = (
        visual
        .analyze_heading_boldness(
            image=image,
            heading_box=
                observations[0]["box"],
            observations=
                observations,
            warning_line_indexes=[
                0,
                1,
            ],
            normal_ratio=1.10,
            bold_ratio=1.25,
        )
    )

    assert (
        result["heading_score"]
        == pytest.approx(
            1.18
        )
    )

    assert (
        result["reference_score"]
        == pytest.approx(
            1.00
        )
    )

    assert (
        result["ratio"]
        == pytest.approx(
            1.18
        )
    )

    assert result["value"] is None

    assert result["reason"] == (
        "boldness_ambiguous"
    )


# ============================================================
# analyze_image semantic fallback
# ============================================================


def test_analyze_image_verifies_when_semantic_model_unavailable(
    monkeypatch,
):
    image = np.zeros(
        (100, 200, 3),
        dtype=np.uint8,
    )

    analysis = {
        "ocr_text":
            "HARBOR STONE",

        "observations": [
            {
                "index": 0,
                "text":
                    "HARBOR STONE",
                "confidence":
                    0.99,
                "box": [
                    10,
                    10,
                    150,
                    30,
                ],
            },
        ],

        "resolved": {
            "brand_name":
                None,
        },

        "unresolved_required_fields":
            [],

        "unresolved_required_roles": [
            "brand_name",
        ],

        "needs_semantics":
            True,

        "government_warning":
            None,
    }

    usability = {
        "usable": True,
    }

    monkeypatch.setattr(
        analyze_image_module,
        "analyze_label_deterministically",
        lambda *args, **kwargs:
            analysis,
    )

    monkeypatch.setattr(
        analyze_image_module,
        "route_analysis",
        lambda analysis: {
            "status":
                "SEMANTIC_FALLBACK",

            "reasons": [
                "semantic_resolution_required",
            ],

            "usability":
                usability,
        },
    )

    def semantic_should_not_run(
        *args,
        **kwargs,
    ):
        pytest.fail(
            "Semantic resolution should not "
            "run without semantic dependencies."
        )

    monkeypatch.setattr(
        analyze_image_module,
        "resolve_semantic_roles",
        semantic_should_not_run,
    )

    result = (
        analyze_image_module
        .analyze_image(
            image=image,
            semantic_model=None,
            semantic_schema=None,
        )
    )

    assert result["status"] == (
        "VERIFY"
    )

    assert result["reasons"] == [
        "semantic_model_unavailable",
    ]

    assert result["usability"] == (
        usability
    )

def test_completes_kentucky_straight_bourbon_whiskey_block():
    observations = [
        {
            "index": 3,
            "text": "KENTUCKY STRAIGHT",
            "confidence": 0.99,
            "box": [
                164,
                874,
                860,
                932,
            ],
        },
        {
            "index": 4,
            "text": "BOURBON WHISKEY",
            "confidence": 0.99,
            "box": [
                79,
                902,
                954,
                1066,
            ],
        },
        {
            "index": 5,
            "text": "SMALLBATCH",
            "confidence": 0.99,
            "box": [
                317,
                1069,
                708,
                1108,
            ],
        },
    ]

    original = {
        "value": "KENTUCKY STRAIGHT",
        "observed_text": "KENTUCKY STRAIGHT",
        "line_indexes": [3],
        "evidence": [
            {
                "line_index": 3,
                "text": "KENTUCKY STRAIGHT",
                "confidence": 0.99,
            }
        ],
        "semantic_observations": [
            "KENTUCKY STRAIGHT",
        ],
        "grounding_method":
            "grounded_semantic_span",
        "method": (
            "grounded_semantic_"
            "class_type_selection"
        ),
    }

    result = (
        complete_class_type_visual_block(
            original,
            observations,
        )
    )

    assert result["value"] == (
        "KENTUCKY STRAIGHT "
        "BOURBON WHISKEY"
    )

    assert result["observed_text"] == (
        "KENTUCKY STRAIGHT\n"
        "BOURBON WHISKEY"
    )

    assert result["line_indexes"] == [
        3,
        4,
    ]

    assert [
        item["text"]
        for item in result["evidence"]
    ] == [
        "KENTUCKY STRAIGHT",
        "BOURBON WHISKEY",
    ]