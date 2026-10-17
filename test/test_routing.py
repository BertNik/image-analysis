import pytest

from lib.routing import (
    assess_ocr_usability,
    route_analysis,
)


def observation(
    text,
    confidence=0.99,
    index=0,
):
    return {
        "index": index,
        "text": text,
        "confidence": confidence,
        "box": [0, 0, 100, 20],
    }


def analysis(
    observations=None,
    missing_fields=None,
    unresolved_roles=None,
):
    return {
        "observations":
            observations or [],

        "unresolved_required_fields":
            missing_fields or [],

        "unresolved_required_roles":
            unresolved_roles or [],
    }


# ============================================================
# assess_ocr_usability
# ============================================================


def test_empty_observations_are_not_usable():
    result = assess_ocr_usability(
        analysis()
    )

    assert result["usable"] is False
    assert result["usable_count"] == 0
    assert result["usable_observations"] == []


@pytest.mark.parametrize(
    "text",
    [
        "",
        ".",
        "..",
        "E",
        "!!",
        " - ",
    ],
)
def test_short_or_meaningless_text_is_ignored(
    text,
):
    result = assess_ocr_usability(
        analysis(
            observations=[
                observation(text),
            ]
        )
    )

    assert result["usable"] is False
    assert result["usable_count"] == 0


def test_three_meaningful_characters_are_usable():
    result = assess_ocr_usability(
        analysis(
            observations=[
                observation(
                    "RUM",
                    confidence=0.99,
                ),
            ]
        )
    )

    assert result["usable"] is True
    assert result["usable_count"] == 1


def test_low_confidence_observation_is_ignored():
    result = assess_ocr_usability(
        analysis(
            observations=[
                observation(
                    "JAMESON",
                    confidence=0.49,
                ),
            ]
        )
    )

    assert result["usable"] is False
    assert result["usable_count"] == 0


def test_confidence_threshold_is_inclusive():
    result = assess_ocr_usability(
        analysis(
            observations=[
                observation(
                    "RUM",
                    confidence=0.50,
                ),
            ]
        )
    )

    assert result["usable"] is True
    assert result["usable_count"] == 1


def test_missing_confidence_does_not_disqualify_text():
    result = assess_ocr_usability(
        analysis(
            observations=[
                observation(
                    "JAMESON",
                    confidence=None,
                ),
            ]
        )
    )

    assert result["usable"] is True
    assert result["usable_count"] == 1


def test_usability_filters_mixed_observations():
    good = observation(
        "JAMESON",
        confidence=0.99,
        index=0,
    )

    too_short = observation(
        "E",
        confidence=0.99,
        index=1,
    )

    too_uncertain = observation(
        "WHISKEY",
        confidence=0.20,
        index=2,
    )

    result = assess_ocr_usability(
        analysis(
            observations=[
                good,
                too_short,
                too_uncertain,
            ]
        )
    )

    assert result["usable"] is True
    assert result["usable_count"] == 1

    assert (
        result["usable_observations"]
        == [good]
    )


def test_custom_minimum_confidence():
    result = assess_ocr_usability(
        analysis(
            observations=[
                observation(
                    "JAMESON",
                    confidence=0.80,
                ),
            ]
        ),
        min_confidence=0.90,
    )

    assert result["usable"] is False


def test_custom_minimum_character_count():
    result = assess_ocr_usability(
        analysis(
            observations=[
                observation(
                    "RUM",
                    confidence=0.99,
                ),
            ]
        ),
        min_meaningful_chars=4,
    )

    assert result["usable"] is False


# ============================================================
# route_analysis
# ============================================================


def test_route_abstains_without_usable_ocr():
    result = route_analysis(
        analysis(
            observations=[
                observation(
                    "..",
                    confidence=0.99,
                ),
                observation(
                    "GARBLED",
                    confidence=0.10,
                    index=1,
                ),
            ]
        )
    )

    assert result["status"] == "ABSTAIN"

    assert result["reasons"] == [
        "no_usable_ocr_evidence"
    ]

    assert (
        result["usability"]["usable"]
        is False
    )


def test_route_verifies_missing_required_field():
    result = route_analysis(
        analysis(
            observations=[
                observation("JAMESON"),
            ],
            missing_fields=[
                "alcohol_content",
            ],
        )
    )

    assert result["status"] == "VERIFY"

    assert result["reasons"] == [
        "alcohol_content_not_resolved"
    ]


def test_route_reports_all_missing_fields():
    result = route_analysis(
        analysis(
            observations=[
                observation("JAMESON"),
            ],
            missing_fields=[
                "brand_name",
                "class_type",
                "alcohol_content",
            ],
        )
    )

    assert result["status"] == "VERIFY"

    assert result["reasons"] == [
        "brand_name_not_resolved",
        "class_type_not_resolved",
        "alcohol_content_not_resolved",
    ]


def test_route_semantic_fallback_for_ambiguous_role():
    result = route_analysis(
        analysis(
            observations=[
                observation(
                    "OLD TOM DISTILLERY"
                ),
            ],
            unresolved_roles=[
                "brand_name",
            ],
        )
    )

    assert (
        result["status"]
        == "SEMANTIC_FALLBACK"
    )

    assert result["reasons"] == [
        "brand_name_role_ambiguous"
    ]


def test_route_reports_multiple_ambiguous_roles():
    result = route_analysis(
        analysis(
            observations=[
                observation(
                    "OLD TOM DISTILLERY"
                ),
            ],
            unresolved_roles=[
                "brand_name",
                "business_name",
            ],
        )
    )

    assert (
        result["status"]
        == "SEMANTIC_FALLBACK"
    )

    assert result["reasons"] == [
        "brand_name_role_ambiguous",
        "business_name_role_ambiguous",
    ]


def test_route_resolves_complete_analysis():
    result = route_analysis(
        analysis(
            observations=[
                observation(
                    "OLD TOM DISTILLERY"
                ),
                observation(
                    "BOURBON WHISKEY",
                    index=1,
                ),
            ]
        )
    )

    assert result["status"] == "RESOLVED"
    assert result["reasons"] == []

    assert (
        result["usability"]["usable"]
        is True
    )


# ============================================================
# Routing precedence
# ============================================================


def test_abstain_has_priority_over_missing_fields():
    """
    If OCR evidence itself is unusable, routing should ABSTAIN
    before considering unresolved fields.
    """

    result = route_analysis(
        analysis(
            observations=[
                observation(
                    "X",
                    confidence=0.99,
                ),
            ],
            missing_fields=[
                "brand_name",
            ],
        )
    )

    assert result["status"] == "ABSTAIN"

    assert result["reasons"] == [
        "no_usable_ocr_evidence"
    ]


def test_missing_fields_have_priority_over_ambiguous_roles():
    """
    Missing deterministic evidence currently takes precedence
    over semantic role ambiguity.
    """

    result = route_analysis(
        analysis(
            observations=[
                observation(
                    "OLD TOM DISTILLERY"
                ),
            ],
            missing_fields=[
                "alcohol_content",
            ],
            unresolved_roles=[
                "brand_name",
            ],
        )
    )

    assert result["status"] == "VERIFY"

    assert result["reasons"] == [
        "alcohol_content_not_resolved"
    ]