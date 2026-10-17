from __future__ import annotations

import re

from typing import Any, Optional
from typing import  Optional
def canonical_field_key(
    field_name: str,
    field: dict[str, Any],
) -> Optional[str]:

    if not field:
        return None

    value = field.get("value")

    if value is None:
        return None

    # Textual fields
    if field_name in {
        "brand_name",
        "class_type",
    }:
        key = normalize_identity(value)

        return key or None

    # Alcohol percentage
    if field_name == "alcohol_content":
        number = numeric_value(field)

        if number is None:
            return None

        return f"percent_abv:{number:.4f}"

    # Proof
    if field_name == "proof":
        number = numeric_value(field)

        if number is None:
            return None

        return f"proof:{number:.4f}"

    # Volume, normalized into milliliters
    if field_name == "net_contents":
        number = numeric_value(field)

        if number is None:
            return None

        unit = str(
            field.get("unit") or ""
        ).strip().lower()

        if unit in {
            "l",
            "liter",
            "liters",
            "litre",
            "litres",
        }:
            milliliters = number * 1000.0

        elif unit in {
            "ml",
            "milliliter",
            "milliliters",
            "millilitre",
            "millilitres",
        }:
            milliliters = number

        else:
            # Do not guess an absent or unknown unit.
            return None

        return f"ml:{milliliters:.4f}"

    return normalize_identity(value) or None


def normalize_identity(
    value: Any,
) -> str:
    """
    Normalize textual fields for comparison across images.

    Example:
        "OLD TOM DISTILLERY"
        "Old Tom Distillery"
        -> OLDTOMDISTILLERY
    """

    return re.sub(
        r"[^A-Z0-9]",
        "",
        str(value or "").upper(),
    )


def numeric_value(
    field: Optional[dict[str, Any]],
) -> Optional[float]:
    if not field:
        return None

    value = field.get("value")

    try:
        return float(value)
    except (TypeError, ValueError):
        return None