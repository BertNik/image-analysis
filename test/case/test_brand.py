from test.helpers.case_fixture import field, usable_image_result
from lib.case.brand import reconcile_brand

def test_single_image_brand_does_not_require_consensus():
    brand = field(
        "RIVEROAK DISTILLING CO.",
        confidence=0.99,
        method=(
            "deterministic_identity_scoring"
        ),
    )

    result = reconcile_brand(
        [
            usable_image_result(
                brand=brand,
            )
        ]
    )

    assert (
        result["status"]
        == "RESOLVED"
    )

    assert (
        result["value"]["value"]
        ==
        "RIVEROAK DISTILLING CO."
    )

    assert (
        result["value"][
            "source_count"
        ]
        == 1
    )

    assert (
        result["value"][
            "reconciliation_method"
        ]
        ==
        "single_image_resolved_brand"
    )


def test_single_image_missing_brand_stays_missing():
    result = reconcile_brand(
        [
            usable_image_result(
                brand=None
            )
        ]
    )

    assert (
        result["status"]
        == "MISSING"
    )


def test_reconcile_brand_empty_input():
    result = (
        reconcile_brand(
            []
        )
    )

    assert (
        result["status"]
        == "MISSING"
    )

