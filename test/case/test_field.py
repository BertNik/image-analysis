import pytest

from lib.case.field import evidence_confidence, add_image_provenance, reconcile_contained_text_groups, reconcile_field
from test.helpers.case_fixture import usable_image_result, field

def test_evidence_confidence_prefers_ocr_confidence():
    assert (
        evidence_confidence(
            {
                "ocr_confidence":
                    0.98,

                "confidence":
                    0.65,
            }
        )
        == pytest.approx(0.98)
    )


def test_evidence_confidence_falls_back_to_confidence():
    assert (
        evidence_confidence(
            {
                "confidence":
                    0.91,
            }
        )
        == pytest.approx(0.91)
    )


def test_evidence_confidence_defaults_to_zero():
    assert (
        evidence_confidence(
            {}
        )
        == 0.0
    )

def test_add_image_provenance():
    original = {
        "value": 750,
        "unit": "mL",
    }

    result = (
        add_image_provenance(
            original,
            {
                "image_index": 2,
                "filename":
                    "back.jpg",
            },
        )
    )

    assert result[
        "image_index"
    ] == 2

    assert result[
        "filename"
    ] == "back.jpg"

    # Original remains unchanged.
    assert (
        "image_index"
        not in original
    )


def test_contained_text_groups_choose_most_specific():
    groups = {
        "BOURBONWHISKEY": [
            {
                "value":
                    "BOURBON WHISKEY"
            }
        ],

        (
            "KENTUCKYSTRAIGHT"
            "BOURBONWHISKEY"
        ): [
            {
                "value":
                    "KENTUCKY STRAIGHT "
                    "BOURBON WHISKEY"
            }
        ],
    }

    result = (
        reconcile_contained_text_groups(
            groups
        )
    )

    assert result is not None

    assert result["key"] == (
        "KENTUCKYSTRAIGHT"
        "BOURBONWHISKEY"
    )


def test_unrelated_text_groups_are_not_compatible():
    result = (
        reconcile_contained_text_groups(
            {
                "BOURBONWHISKEY": [
                    {}
                ],
                "VODKA": [
                    {}
                ],
            }
        )
    )

    assert result is None


def test_empty_text_groups_return_none():
    assert (
        reconcile_contained_text_groups(
            {}
        )
        is None
    )


# ============================================================
# Basic field reconciliation
# ============================================================


def test_reconcile_field_missing():
    result = reconcile_field(
        "alcohol_content",
        [
            usable_image_result()
        ],
    )

    assert (
        result["status"]
        == "MISSING"
    )

    assert result["value"] is None


def test_reconcile_equal_volume_across_images():
    results = [
        usable_image_result(
            image_index=0,
            net_contents=field(
                750,
                unit="mL",
                confidence=0.90,
            ),
        ),

        usable_image_result(
            image_index=1,
            net_contents=field(
                750,
                unit="mL",
                confidence=0.99,
            ),
        ),
    ]

    result = reconcile_field(
        "net_contents",
        results,
    )

    assert (
        result["status"]
        == "RESOLVED"
    )

    assert (
        result["value"]["value"]
        == 750
    )

    assert (
        result["value"][
            "source_count"
        ]
        == 2
    )

    # Highest-confidence agreeing
    # observation is representative.
    assert (
        result["value"][
            "image_index"
        ]
        == 1
    )


def test_reconcile_equivalent_liter_and_ml():
    results = [
        usable_image_result(
            image_index=0,
            net_contents=field(
                750,
                unit="mL",
            ),
        ),

        usable_image_result(
            image_index=1,
            net_contents=field(
                0.75,
                unit="L",
            ),
        ),
    ]

    result = reconcile_field(
        "net_contents",
        results,
    )

    assert (
        result["status"]
        == "RESOLVED"
    )

    assert (
        result["value"][
            "source_count"
        ]
        == 2
    )


def test_conflicting_abv_remains_conflict():
    results = [
        usable_image_result(
            image_index=0,
            alcohol=field(
                40,
                unit="percent_abv",
            ),
        ),

        usable_image_result(
            image_index=1,
            alcohol=field(
                45,
                unit="percent_abv",
            ),
        ),
    ]

    result = reconcile_field(
        "alcohol_content",
        results,
    )

    assert (
        result["status"]
        == "CONFLICT"
    )

    assert result["value"] is None

    assert (
        len(result["conflicts"])
        == 2
    )



def test_class_type_partial_and_specific_are_compatible():
    results = [
        usable_image_result(
            image_index=0,
            class_type=field(
                "BOURBON WHISKEY"
            ),
        ),

        usable_image_result(
            image_index=1,
            class_type=field(
                "KENTUCKY STRAIGHT "
                "BOURBON WHISKEY"
            ),
        ),
    ]

    result = reconcile_field(
        "class_type",
        results,
    )

    assert (
        result["status"]
        == "RESOLVED"
    )

    assert (
        result["value"]["value"]
        ==
        "KENTUCKY STRAIGHT "
        "BOURBON WHISKEY"
    )

    assert (
        result["value"][
            "reconciliation_method"
        ]
        ==
        "most_specific_compatible_value"
    )


def test_unrelated_class_types_conflict():
    results = [
        usable_image_result(
            image_index=0,
            class_type=field(
                "BOURBON WHISKEY"
            ),
        ),

        usable_image_result(
            image_index=1,
            class_type=field(
                "VODKA"
            ),
        ),
    ]

    result = reconcile_field(
        "class_type",
        results,
    )

    assert (
        result["status"]
        == "CONFLICT"
    )
