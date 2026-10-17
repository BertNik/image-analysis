import cv2
import numpy as np
import pytest

from lib.resolver.visual import analyze_heading_boldness
from lib.warning_label import (
        validate_government_warning, 
        extract_government_warning, 
        GOVERNMENT_WARNING_TEXT
    )

def make_warning_boldness_fixture(
    *,
    heading_thickness: int,
    body_thickness: int,
):
    image = np.full(
        (300, 900, 3),
        255,
        dtype=np.uint8,
    )

    font_face = (
        cv2.FONT_HERSHEY_SIMPLEX
    )

    heading_text = (
        "GOVERNMENT WARNING:"
    )

    heading_origin = (
        30,
        70,
    )

    heading_scale = 0.8

    (
        heading_size,
        heading_baseline,
    ) = cv2.getTextSize(
        heading_text,
        font_face,
        heading_scale,
        heading_thickness,
    )

    cv2.putText(
        image,
        heading_text,
        heading_origin,
        font_face,
        heading_scale,
        (0, 0, 0),
        heading_thickness,
        cv2.LINE_AA,
    )

    heading_x, heading_y = (
        heading_origin
    )

    heading_box = [
        heading_x - 4,
        heading_y
        - heading_size[1]
        - 4,
        heading_x
        + heading_size[0]
        + 4,
        heading_y
        + heading_baseline
        + 4,
    ]

    observations = [
        {
            "index": 0,
            "text":
                heading_text,
            "box":
                heading_box,
        },
    ]

    body_lines = [
        (
            "(1) According to the "
            "Surgeon General"
        ),
        (
            "women should not drink "
            "alcoholic beverages"
        ),
        (
            "during pregnancy because "
            "of the risk"
        ),
        "of birth defects.",
    ]

    body_scale = 0.55
    body_x = 30
    body_y = 125

    for index, text in enumerate(
        body_lines,
        start=1,
    ):
        (
            text_size,
            baseline,
        ) = cv2.getTextSize(
            text,
            font_face,
            body_scale,
            body_thickness,
        )

        cv2.putText(
            image,
            text,
            (
                body_x,
                body_y,
            ),
            font_face,
            body_scale,
            (0, 0, 0),
            body_thickness,
            cv2.LINE_AA,
        )

        box = [
            body_x - 4,
            body_y
            - text_size[1]
            - 4,
            body_x
            + text_size[0]
            + 4,
            body_y
            + baseline
            + 4,
        ]

        observations.append(
            {
                "index":
                    index,

                "text":
                    text,

                "box":
                    box,
            }
        )

        body_y += 38

    return image, observations

def test_warning_extraction_returns_none_when_heading_box_invalid():
    observations = [
        {
            "index": 0,
            "text":
                "GOVERNMENT WARNING:",
            "confidence":
                0.99,
            "box": [
                100,
                200,
            ],
        },
    ]

    result = (
        extract_government_warning(
            observations
        )
    )

    assert result is None

def test_warning_extraction_skips_empty_lines():
    observations = [
        {
            "index": 0,
            "text":
                "GOVERNMENT WARNING:",
            "confidence":
                0.99,
            "box": [
                100,
                100,
                300,
                120,
            ],
        },
        {
            "index": 1,
            "text":
                "   ",
            "confidence":
                0.99,
            "box": [
                100,
                125,
                300,
                145,
            ],
        },
        {
            "index": 2,
            "text":
                "HEALTH PROBLEMS.",
            "confidence":
                0.98,
            "box": [
                100,
                150,
                300,
                170,
            ],
        },
    ]

    result = (
        extract_government_warning(
            observations
        )
    )

    assert result is not None

    assert 1 not in (
        result["line_indexes"]
    )

    assert result["line_indexes"] == [
        0,
        2,
    ]

def test_warning_extraction_skips_other_column():
    observations = [
        {
            "index": 0,
            "text":
                "GOVERNMENT WARNING:",
            "confidence":
                0.99,
            "box": [
                600,
                100,
                800,
                120,
            ],
        },
        {
            "index": 1,
            "text":
                "750 mL",
            "confidence":
                0.99,
            "box": [
                100,
                130,
                200,
                150,
            ],
        },
        {
            "index": 2,
            "text":
                "(1) According to the Surgeon General,",
            "confidence":
                0.98,
            "box": [
                600,
                130,
                900,
                150,
            ],
        },
    ]

    result = (
        extract_government_warning(
            observations
        )
    )

    assert result is not None

    assert result["line_indexes"] == [
        0,
        2,
    ]

    assert (
        "750 mL"
        not in result[
            "observed_text"
        ]
    )

def test_warning_extraction_detects_end_text_across_lines():
    observations = [
        {
            "index": 0,
            "text":
                "GOVERNMENT WARNING:",
            "confidence":
                0.99,
            "box": [
                600,
                100,
                800,
                120,
            ],
        },
        {
            "index": 1,
            "text":
                "OPERATE MACHINERY, AND MAY CAUSE",
            "confidence":
                0.98,
            "box": [
                600,
                130,
                900,
                150,
            ],
        },
        {
            "index": 2,
            "text":
                "HEALTH PROBLEMS.",
            "confidence":
                0.97,
            "box": [
                600,
                160,
                800,
                180,
            ],
        },
        {
            "index": 3,
            "text":
                "PRODUCT OF USA",
            "confidence":
                0.99,
            "box": [
                600,
                190,
                800,
                210,
            ],
        },
    ]

    result = (
        extract_government_warning(
            observations
        )
    )

    assert result is not None

    assert result["complete"] is True

    assert result["line_indexes"] == [
        0,
        1,
        2,
    ]

    assert (
        "PRODUCT OF USA"
        not in result[
            "observed_text"
        ]
    )

def test_warning_extraction_remains_incomplete_without_terminal_text():
    observations = [
        {
            "index": 0,
            "text":
                "GOVERNMENT WARNING:",
            "confidence":
                0.99,
            "box": [
                600,
                100,
                800,
                120,
            ],
        },
        {
            "index": 1,
            "text":
                "(1) According to the Surgeon General,",
            "confidence":
                0.95,
            "box": [
                600,
                130,
                900,
                150,
            ],
        },
        {
            "index": 2,
            "text":
                "women should not drink alcoholic beverages",
            "confidence":
                0.90,
            "box": [
                600,
                160,
                900,
                180,
            ],
        },
    ]

    result = (
        extract_government_warning(
            observations
        )
    )

    assert result is not None
    assert result["complete"] is False

    assert result["line_indexes"] == [
        0,
        1,
        2,
    ]

def test_warning_extraction_uses_lowest_warning_confidence():
    observations = [
        {
            "index": 0,
            "text":
                "GOVERNMENT WARNING:",
            "confidence":
                0.99,
            "box": [
                600,
                100,
                800,
                120,
            ],
        },
        {
            "index": 1,
            "text":
                "OPERATE MACHINERY, AND MAY CAUSE",
            "confidence":
                0.85,
            "box": [
                600,
                130,
                900,
                150,
            ],
        },
        {
            "index": 2,
            "text":
                "HEALTH PROBLEMS.",
            "confidence":
                0.97,
            "box": [
                600,
                160,
                800,
                180,
            ],
        },
    ]

    result = (
        extract_government_warning(
            observations
        )
    )

    assert result is not None
    assert result["confidence"] == (
        pytest.approx(
            0.85
        )
    )

def test_warning_extraction_returns_none_when_heading_box_missing():
    observations = [
        {
            "index": 0,
            "text":
                "GOVERNMENT WARNING:",
            "confidence":
                0.99,
            "box":
                None,
        },
    ]

    result = (
        extract_government_warning(
            observations
        )
    )

    assert result is None

def test_bold_warning_heading_is_heavier_than_body():
    image, observations = (
        make_warning_boldness_fixture(
            heading_thickness=3,
            body_thickness=1,
        )
    )

    result = analyze_heading_boldness(
        image=image,
        heading_box=
            observations[0]["box"],
        observations=
            observations,
        warning_line_indexes=[
            0,
            1,
            2,
            3,
            4,
        ],
    )

    assert (
        result["heading_score"]
        is not None
    )

    assert (
        result["reference_score"]
        is not None
    )

    assert result["reference_count"] == 4

    assert (
        result["heading_score"]
        > result["reference_score"]
    )

    assert result["ratio"] > 1.0
    assert result["value"] is True

    assert result["method"] == (
        "relative_warning_stroke_width"
    )

def test_warning_body_case_differences_are_accepted():
    warning = {
        "observed_text": (
            GOVERNMENT_WARNING_TEXT.upper()
        ),
        "heading_text":
            "GOVERNMENT WARNING:",
        "complete":
            True,
        "heading_bold":
            True,
    }

    result = validate_government_warning(
        warning
    )

    assert result["text_exact"] is True
    assert result["heading_uppercase"] is True
    assert result["heading_bold"] is True
    assert result["status"] == "PASS"

def test_regular_warning_heading_matches_body_weight():
    image, observations = (
       make_warning_boldness_fixture(
            heading_thickness=1,
            body_thickness=1,
       )
    )

    result = analyze_heading_boldness(
        image=image,
        heading_box=
           observations[0]["box"],
        observations=
            observations,
        warning_line_indexes=[
            0,
            1,
            2,
            3,
            4,
        ],
    )

    assert (
       result["heading_score"]
       is not None
    )

    assert (
        result["reference_score"] is not None
    )

    assert (
        result["reference_count"]
        == 4
    )

    assert result["ratio"] == (
        pytest.approx(
            1.0,
           abs=0.25,
        )
    )

    assert result["value"] is False

    assert result["method"] == (
       "relative_warning_stroke_width"
    )

def test_bold_heading_ratio_exceeds_regular_heading_ratio():
    (
        bold_image,
        bold_observations,
    ) = make_warning_boldness_fixture(
        heading_thickness=3,
        body_thickness=1,
    )

    (
        regular_image,
        regular_observations,
    ) = make_warning_boldness_fixture(
        heading_thickness=1,
        body_thickness=1,
    )
    

    bold_result = (
        analyze_heading_boldness(
            image=bold_image,
            heading_box=(
                bold_observations[0][
                    "box"
                ]
            ),
           observations=(
               bold_observations
           ),
            warning_line_indexes=[
                0,
                1,
                2,
                3,
                4,
            ],
        )
    )

    regular_result = (
        analyze_heading_boldness(
            image=regular_image,
            heading_box=(
                regular_observations[0][
                    "box"
                ]
            ),
           observations=(
               regular_observations
           ),
            warning_line_indexes=[
                0,
                1,
                2,
                3,
                4,
            ],
        )
    )

    assert (
        bold_result["ratio"]
        is not None
    )

    assert (
        regular_result["ratio"]
        is not None
    )

    assert (
        bold_result["ratio"]
        > regular_result["ratio"]
    )

    assert bold_result["value"] is True
    assert regular_result["value"] is False


def test_warning_extraction_returns_none_without_heading():
    observations = [
        {
            "index": 0,
            "text":
                "KENTUCKY STRAIGHT",
            "confidence":
                0.99,
            "box": [
                100,
                100,
                300,
                120,
            ],
        },
        {
            "index": 1,
            "text":
                "BOURBON WHISKEY",
            "confidence":
                0.99,
            "box": [
                100,
                130,
                300,
                150,
            ],
        },
    ]

    result = (
        extract_government_warning(
            observations
        )
    )

    assert result is None


@pytest.mark.parametrize(
    (
        "changes",
        "expected_status",
        "expected_reason",
    ),
    [
        # --------------------------------------------
        # Incomplete warning
        # --------------------------------------------
        (
            {
                "complete": False,
            },
            "FAIL",
            "government_warning_incomplete",
        ),

        # --------------------------------------------
        # Text mismatch
        # --------------------------------------------
        (
            {
                "observed_text": (
                    GOVERNMENT_WARNING_TEXT.replace(
                        "alcoholic",
                        "alcoholid",
                        1,
                    )
                ),
            },
            "FAIL",
            "government_warning_text_mismatch",
        ),

        # --------------------------------------------
        # Heading is not uppercase
        # --------------------------------------------
        (
            {
                "heading_text":
                    "Government Warning:",
            },
            "FAIL",
            (
                "government_warning_"
                "heading_not_uppercase"
            ),
        ),

        # --------------------------------------------
        # Heading explicitly established as not bold
        # --------------------------------------------
        (
            {
                "heading_bold": False,
            },
            "FAIL",
            (
                "government_warning_"
                "heading_not_bold"
            ),
        ),

        # --------------------------------------------
        # Everything passes except visual boldness
        # cannot be established
        # --------------------------------------------
        (
            {
                "heading_bold": None,
            },
            "VERIFY",
            (
                "government_warning_"
                "boldness_unresolved"
            ),
        ),
    ],
)

def test_government_warning_failure_and_verify_paths(
    changes,
    expected_status,
    expected_reason,
):
    warning = {
        "observed_text":
            GOVERNMENT_WARNING_TEXT,

        "heading_text":
            "GOVERNMENT WARNING:",

        "complete":
            True,

        # Default passing visual state.
        "heading_bold":
            True,
    }

    warning.update(
        changes
    )

    result = (
        validate_government_warning(
            warning
        )
    )

    assert (
        result["status"]
        == expected_status
    )

    assert (
        result["reason"]
        == expected_reason
    )

def test_boldness_uses_warning_body_as_reference():
    image,observations = (
        make_warning_boldness_fixture(
            heading_thickness=3,
            body_thickness=1,
        )
    )
    result = analyze_heading_boldness(
        image=image,
        heading_box=
            observations[0]["box"],
        observations=
            observations,
        warning_line_indexes=[
            0,
            1,
            2,
            3,
            4,
        ],
    )

    assert (
        result["reference_count"]
        == 4
    )

    reference_indexes = {
        item["index"]
        for item
        in result[
            "reference_evidence"
        ]
    }

    assert reference_indexes == {
        1,
        2,
        3,
        4,
    }

    assert 0 not in reference_indexes

    assert result["method"] == (
        "relative_warning_stroke_width"
    )

def test_boldness_is_unresolved_without_warning_body():
    image, observations = (
        make_warning_boldness_fixture(
            heading_thickness=3,
            body_thickness=1,
        )
    )

    result = analyze_heading_boldness(
        image=image,
        heading_box=
            observations[0]["box"],
        observations=
            observations,
        warning_line_indexes=[0],
    )

    assert (
        result["heading_score"]
        is not None
    )

    assert (
        result["reference_score"]
        is None
    )

    assert result["ratio"] is None
    assert result["value"] is None

    assert (
        result["reference_count"]
        == 0
    )

    assert result["reason"] == (
        "no_warning_body_reference"
    )

def make_boldness_test_image(
    *,
    heading_thickness: int,
    reference_thickness: int,
):
    image = np.full(
        (220, 800, 3),
        255,
        dtype=np.uint8,
    )

    # Candidate warning heading.
    cv2.putText(
        image,
        "GOVERNMENT WARNING:",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        heading_thickness,
        cv2.LINE_AA,
    )

    # Similar-sized reference text.
    cv2.putText(
        image,
        "REFERENCE TEXT",
        (20, 165),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        reference_thickness,
        cv2.LINE_AA,
    )

    observations = [
        {
            "index": 0,
            "text":
                "GOVERNMENT WARNING:",
            "box": [
                15,
                40,
                390,
                90,
            ],
        },
        {
            "index": 1,
            "text":
                "REFERENCE TEXT",
            "box": [
                15,
                130,
                300,
                180,
            ],
        },
    ]

    return image, observations