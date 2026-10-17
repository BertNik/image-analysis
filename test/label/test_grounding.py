import pytest

from lib.label.grounding import (
    ground_semantic_span,
    identity_key
)

@pytest.mark.parametrize(
    "value, expected",
    [
        (
            None,
            "",
        ),
        (
            "",
            "",
        ),
        (
            "Captain Morgan",
            "CAPTAINMORGAN",
        ),
        (
            "Stone's Throw",
            "STONESTHROW",
        ),
        (
            "RIVER OAK DISTILLING CO.",
            "RIVEROAKDISTILLINGCO",
        ),
        (
            "  Old-Tom & Co.  ",
            "OLDTOMCO",
        ),
    ],
)
def test_identity_key(
    value,
    expected,
):
    assert (
        identity_key(value)
        == expected
    )

def observation(
    index,
    text,
    confidence=0.99,
):
    return {
        "index": index,
        "text": text,
        "confidence": confidence,
        "box": None,
    }


def test_ground_exact_brand():
    result = ground_semantic_span(
        "JAMESON",
        [
            observation(
                0,
                "JAMESON",
            )
        ],
    )

    assert result is not None
    assert result["value"] == "JAMESON"
    assert result["line_indexes"] == [0]


def test_ground_normalized_brand():
    result = ground_semantic_span(
        "Captain Morgan",
        [
            observation(
                3,
                "CaptainMorgan",
            )
        ],
    )

    assert result is not None

    assert (
        result["value"]
        == "CaptainMorgan"
    )


def test_semantic_model_cannot_invent_brand():
    result = ground_semantic_span(
        "INVENTED BRAND",
        [
            observation(
                0,
                "JAMESON",
            )
        ],
    )

    assert result is None


def test_empty_semantic_brand_is_rejected():
    result = ground_semantic_span(
        "",
        [
            observation(
                0,
                "JAMESON",
            )
        ],
    )

    assert result is None


def test_repeated_brand_occurrences_preserve_provenance():
    observations = [
        {
            "index": 3,
            "text": "CaptainMorgan",
            "confidence": 0.95,
            "box": None,
        },
        {
            "index": 7,
            "text": "CaptainMorgan",
            "confidence": 0.99,
            "box": None,
        },
    ]

    result = ground_semantic_span(
        "Captain Morgan",
        observations,
    )

    assert result is not None

    assert (
        result["value"]
        == "CaptainMorgan"
    )

    assert result["line_indexes"] == [
        3,
        7,
    ]

    assert result["evidence"] == [
        {
            "line_index": 3,
            "text": "CaptainMorgan",
            "confidence": 0.95,
        },
        {
            "line_index": 7,
            "text": "CaptainMorgan",
            "confidence": 0.99,
        },
    ]

    assert (
        result["method"]
        == "grounded_semantic_span"
    )