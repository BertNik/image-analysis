import pytest
import cv2
import numpy as np

from lib.label.analyzer import (
    analyze_label_deterministically,
    build_observations,
)

from lib.label.measurements import (
    extract_measurements,
    resolve_proof_constraint,
)

def make_ocr(
    lines,
    scores=None,
    boxes=None,
):
    """
    Build an OCR result using the structure expected by
    analyze_label_deterministically().
    """

    if scores is None:
        scores = [
            0.99
            for _ in lines
        ]

    if boxes is None:
        boxes = [
            [
                100,
                100 + index * 50,
                500,
                140 + index * 50,
            ]
            for index in range(
                len(lines)
            )
        ]

    return {
        "text": "\n".join(lines),
        "lines": lines,
        "scores": scores,
        "boxes": boxes,
    }


# ============================================================
# Measurements
# ============================================================


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "45% Alc./Vol.",
            45.0,
        ),
        (
            "45% ALC./VOL.",
            45.0,
        ),
        (
            "35% alc./vol.",
            35.0,
        ),
        (
            "21% Alc/Vol",
            21.0,
        ),
    ],
)
def test_abv_regression(
    text,
    expected,
):
    observations = build_observations(
        make_ocr(
            [
                text,
            ]
        )
    )

    result = extract_measurements(
        observations
    )

    alcohol = result[
        "alcohol_content"
    ]

    assert alcohol is not None

    assert (
        alcohol["value"]
        == expected
    )

    assert (
        alcohol["unit"]
        == "percent_abv"
    )


@pytest.mark.parametrize(
    "text, value, unit",
    [
        (
            "200 ml",
            200.0,
            "mL",
        ),
        (
            "750 mL",
            750.0,
            "mL",
        ),
        (
            "750 mL.",
            750.0,
            "mL",
        ),
        (
            "1 L",
            1.0,
            "L",
        ),
    ],
)
def test_net_contents_regression(
    text,
    value,
    unit,
):
    observations = build_observations(
        make_ocr(
            [
                text,
            ]
        )
    )

    result = extract_measurements(
        observations
    )

    net = result[
        "net_contents"
    ]

    assert net is not None

    assert (
        net["value"]
        == value
    )

    assert (
        net["unit"]
        == unit
    )


def test_exact_proof_regression():
    observations = build_observations(
        make_ocr(
            [
                "(90 PROOF)",
            ]
        )
    )

    result = extract_measurements(
        observations
    )

    proof = result["proof"]

    assert proof is not None

    assert (
        proof["value"]
        == 90.0
    )

    assert (
        proof["unit"]
        == "proof"
    )


def test_corrupt_proof_can_be_resolved_by_abv_constraint():
    observations = build_observations(
        make_ocr(
            [
                "45% ALC./VOL.",
                "(90 PR00F)",
            ],
            scores=[
                0.99,
                0.93,
            ],
        )
    )

    measurements = extract_measurements(
        observations
    )

    assert (
        measurements["proof"]
        is None
    )

    assert (
        len(
            measurements[
                "unresolved_measurements"
            ]
        )
        == 1
    )

    resolved = resolve_proof_constraint(
        measurements
    )

    assert resolved is not None

    assert (
        resolved["value"]
        == 90.0
    )

    assert (
        resolved["method"]
        == "cross_field_constraint"
    )


def test_proof_constraint_does_not_accept_wrong_value():
    observations = build_observations(
        make_ocr(
            [
                "40% ALC./VOL.",
                "(90 PR00F)",
            ]
        )
    )

    measurements = extract_measurements(
        observations
    )

    resolved = resolve_proof_constraint(
        measurements
    )

    assert resolved is None


# ============================================================
# Semantic handoff
# ============================================================


def test_brand_and_class_type_are_left_for_semantic_resolution():
    result = (
        analyze_label_deterministically(
            make_ocr(
                [
                    "JAMESON",
                    "IRISH WHISKEY",
                    "40% ALC./VOL.",
                    "750 mL",
                ]
            ),
            brand_required=True,
        )
    )

    resolved = result[
        "resolved"
    ]

    # Brand role is semantic.
    assert (
        resolved["brand_name"]
        is None
    )

    # Class/type role is semantic.
    assert (
        resolved["class_type"]
        is None
    )

    assert set(
        result[
            "unresolved_required_roles"
        ]
    ) == {
        "brand_name",
        "class_type",
    }

    assert (
        result["unresolved"][
            "brand_name"
        ]["reason"]
        ==
        "brand_requires_semantic_resolution"
    )

    assert (
        result["unresolved"][
            "class_type"
        ]["reason"]
        ==
        "class_type_requires_semantic_resolution"
    )

    # Explicit measurements remain deterministic.
    assert (
        resolved[
            "alcohol_content"
        ]["value"]
        == 40.0
    )

    assert (
        resolved[
            "net_contents"
        ]["value"]
        == 750.0
    )

    assert (
        result["needs_semantics"]
        is True
    )

    assert (
        result[
            "deterministic_complete"
        ]
        is False
    )

def test_brand_not_required_still_requires_class_type_semantics():
    result = (
        analyze_label_deterministically(
            make_ocr(
                [
                    "IRISH WHISKEY",
                    "40% ALC./VOL.",
                    "750 mL",
                ]
            ),
            brand_required=False,
        )
    )

    assert (
        result["resolved"][
            "brand_name"
        ]
        is None
    )

    assert (
        result["resolved"][
            "class_type"
        ]
        is None
    )

    assert (
        result[
            "unresolved_required_roles"
        ]
        == [
            "class_type",
        ]
    )

    assert (
        "brand_name"
        not in result["unresolved"]
    )

    assert (
        result["unresolved"][
            "class_type"
        ]["reason"]
        ==
        "class_type_requires_semantic_resolution"
    )

    assert (
        result["needs_semantics"]
        is True
    )


def test_missing_alcohol_content_remains_required_field():
    result = (
        analyze_label_deterministically(
            make_ocr(
                [
                    "JAMESON",
                    "IRISH WHISKEY",
                    "750 mL",
                ]
            ),
            brand_required=True,
        )
    )

    assert (
        result["resolved"][
            "alcohol_content"
        ]
        is None
    )

    assert (
        "alcohol_content"
        in result[
            "unresolved_required_fields"
        ]
    )

    assert (
        result["unresolved"][
            "alcohol_content"
        ]["reason"]
        ==
        "alcohol_content_not_observed"
    )

    assert set(
        result[
            "unresolved_required_roles"
        ]
    ) == {
        "brand_name",
        "class_type",
    }


def test_missing_net_contents_is_not_observed_not_required_unresolved():
    result = (
        analyze_label_deterministically(
            make_ocr(
                [
                    "JAMESON",
                    "IRISH WHISKEY",
                    "40% ALC./VOL.",
                ]
            ),
            brand_required=True,
        )
    )

    assert (
        result["resolved"][
            "net_contents"
        ]
        is None
    )

    assert (
        "net_contents"
        in result[
            "not_observed_fields"
        ]
    )

    assert (
        "net_contents"
        not in result[
            "unresolved_required_fields"
        ]
    )

# ============================================================
# Full deterministic contract
# ============================================================


def test_river_oak_deterministic_handoff_regression():
    lines = [
        "EST.1870",
        "RIVEROAK",
        "DISTILLING CO.",
        "KENTUCKY STRAIGHT",
        "BOURBON WHISKEY",
        "SMALLBATCH",
        "45% ALC./VOL.",
        "750 mL",
        "(90 PR00F)",
        "DISTILLED AND BOTTLED BY",
        "RIVER OAK DISTILLING CO.",
        "FRANKFORT, KY 40601",
    ]

    scores = [
        0.9998,
        0.9996,
        0.9996,
        0.9907,
        0.9999,
        0.9999,
        0.9999,
        0.9623,
        0.9278,
        0.9735,
        0.9968,
        0.9827,
    ]

    boxes = [
        [370, 125, 648, 166],
        [7, 133, 1022, 551],
        [229, 448, 809, 524],
        [164, 874, 860, 932],
        [79, 902, 954, 1066],
        [317, 1069, 708, 1108],
        [98, 1129, 337, 1212],
        [784, 1128, 911, 1205],
        [424, 1158, 652, 1216],
        [314, 1252, 723, 1290],
        [257, 1285, 769, 1331],
        [338, 1331, 691, 1365],
    ]

    result = (
        analyze_label_deterministically(
            make_ocr(
                lines,
                scores=scores,
                boxes=boxes,
            ),
            brand_required=True,
        )
    )

    resolved = result[
        "resolved"
    ]

    # Explicit values remain deterministic.
    assert (
        resolved[
            "alcohol_content"
        ]["value"]
        == 45.0
    )

    assert (
        resolved["proof"]["value"]
        == 90.0
    )

    assert (
        resolved[
            "net_contents"
        ]["value"]
        == 750.0
    )

    # Semantic roles are deliberately not assigned here.
    assert (
        resolved["brand_name"]
        is None
    )

    assert (
        resolved["class_type"]
        is None
    )

    assert set(
        result[
            "unresolved_required_roles"
        ]
    ) == {
        "brand_name",
        "class_type",
    }

    assert (
        result["unresolved"][
            "brand_name"
        ]["reason"]
        ==
        "brand_requires_semantic_resolution"
    )

    assert (
        result["unresolved"][
            "class_type"
        ]["reason"]
        ==
        "class_type_requires_semantic_resolution"
    )

    assert result[
        "needs_semantics"
    ] is True

    assert result[
        "deterministic_complete"
    ] is False

    # resolution. The deterministic analyzer must not assign
    # business names, roles, addresses, or relationships.
    assert (
        resolved["businesses"]
        == []
    )