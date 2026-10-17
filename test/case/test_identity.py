import pytest
from lib.case.identity import (normalize_identity, numeric_value, canonical_field_key)


@pytest.mark.parametrize(
    "value, expected",
    [
        (
            "OLD TOM DISTILLERY",
            "OLDTOMDISTILLERY",
        ),
        (
            "Old-Tom Distillery",
            "OLDTOMDISTILLERY",
        ),
        (
            "Stone's Throw",
            "STONESTHROW",
        ),
        (
            None,
            "",
        ),
    ],
)
def test_normalize_identity(
    value,
    expected,
):
    assert (
        normalize_identity(
            value
        )
        == expected
    )


@pytest.mark.parametrize(
    "input_field, expected",
    [
        (
            {
                "value": 45,
            },
            45.0,
        ),
        (
            {
                "value": "45.5",
            },
            45.5,
        ),
        (
            {
                "value": "bad",
            },
            None,
        ),
        (
            None,
            None,
        ),
    ],
)
def test_numeric_value(
    input_field,
    expected,
):
    assert (
        numeric_value(
            input_field
        )
        == expected
    )


# ============================================================
# Canonical field keys
# ============================================================


def test_canonical_brand_key():
    assert (
        canonical_field_key(
            "brand_name",
            {
                "value":
                    "Old Tom Distillery",
            },
        )
        == "OLDTOMDISTILLERY"
    )


def test_canonical_alcohol_key():
    assert (
        canonical_field_key(
            "alcohol_content",
            {
                "value": 45,
                "unit":
                    "percent_abv",
            },
        )
        == "percent_abv:45.0000"
    )


def test_canonical_proof_key():
    assert (
        canonical_field_key(
            "proof",
            {
                "value": 90,
                "unit": "proof",
            },
        )
        == "proof:90.0000"
    )


def test_canonical_volume_ml():
    assert (
        canonical_field_key(
            "net_contents",
            {
                "value": 750,
                "unit": "mL",
            },
        )
        == "ml:750.0000"
    )


def test_canonical_volume_liters():
    assert (
        canonical_field_key(
            "net_contents",
            {
                "value": 0.75,
                "unit": "L",
            },
        )
        == "ml:750.0000"
    )


def test_unknown_volume_unit_is_not_guessed():
    assert (
        canonical_field_key(
            "net_contents",
            {
                "value": 750,
                "unit": "ounces",
            },
        )
        is None
    )