import pytest

from test.helpers.case_fixture import complete_image
from lib.case.reconcile import reconcile_case
from lib.case.constants import REQUIRED_CASE_FIELDS

def test_reconcile_case_with_no_images_abstains():
    result = reconcile_case(
        []
    )

    assert (
        result["status"]
        == "ABSTAIN"
    )

    assert result["reasons"] == [
        "no_images_processed"
    ]

    assert (
        set(
            result["analysis"][
                "unresolved_required_fields"
            ]
        )
        ==
        set(
            REQUIRED_CASE_FIELDS
        )
    )


def test_reconcile_case_with_no_usable_images_abstains():
    result = reconcile_case(
        [
            {
                "image_index": 0,
                "filename":
                    "bad.jpg",

                "status": "ERROR",

                "error":
                    "decode_failed",

                "usability": {
                    "usable": False,
                },
            }
        ]
    )

    assert (
        result["status"]
        == "ABSTAIN"
    )

    assert result["reasons"] == [
        "no_usable_image_evidence"
    ]


def test_complete_single_image_case_resolves():
    result = (
        reconcile_case(
            [
                complete_image()
            ]
        )
    )

    assert (
        result["status"]
        == "RESOLVED"
    )

    assert result["reasons"] == []

    assert (
        result["analysis"][
            "resolved"
        ]["brand_name"]["value"]
        == "OLD TOM DISTILLERY"
    )

    assert (
        result["analysis"][
            "analysis_complete"
        ]
        is True
    )


def test_required_missing_field_routes_case_to_verify():
    image = complete_image()

    image["analysis"][
        "resolved"
    ]["alcohol_content"] = None

    result = (
        reconcile_case(
            [image]
        )
    )

    assert (
        result["status"]
        == "VERIFY"
    )

    assert (
        "alcohol_content_not_resolved"
        in result["reasons"]
    )

    assert (
        "alcohol_content"
        in result["analysis"][
            "unresolved_required_fields"
        ]
    )


def test_missing_optional_proof_does_not_fail_case():
    image = complete_image()

    image["analysis"][
        "resolved"
    ]["proof"] = None

    result = (
        reconcile_case(
            [image]
        )
    )

    assert (
        result["status"]
        == "RESOLVED"
    )

    assert (
        "proof"
        in result["analysis"][
            "not_observed_fields"
        ]
    )

def test_case_collects_semantic_fields():
    first = complete_image(
        image_index=0
    )

    second = complete_image(
        image_index=1
    )

    first["analysis"][
        "semantic_fields"
    ] = [
        "brand_name"
    ]

    second["analysis"][
        "semantic_fields"
    ] = [
        "brand_name",
        "class_type",
    ]

    result = (
        reconcile_case(
            [
                first,
                second,
            ]
        )
    )

    assert (
        result["analysis"][
            "semantic_fields"
        ]
        == [
            "brand_name",
            "class_type",
        ]
    )

def test_case_sums_image_processing_times():
    first = complete_image(
        image_index=0
    )

    second = complete_image(
        image_index=1
    )

    first["timing"][
        "total_seconds"
    ] = 0.25

    second["timing"][
        "total_seconds"
    ] = 0.40

    result = (
        reconcile_case(
            [
                first,
                second,
            ]
        )
    )

    assert (
        result["timing"][
            "total_seconds"
        ]
        == pytest.approx(0.65)
    )

    assert (
        result["timing"][
            "image_count"
        ]
        == 2
    )

    assert (
        result["timing"][
            "usable_image_count"
        ]
        == 2
    )
