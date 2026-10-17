from __future__ import annotations

from .extraction import (
    ensure_extraction,
)
from .grounding import (
    collect_grounded_entities,
    format_grounded_entity
)

BRAND_EXCLUDED_TOKENS = {
    "ALC",
    "ALCOHOL",
    "PROOF",
    "ML",
    "L",
    "WARNING",
    "GOVERNMENT",
}

BRAND_FIELD_LABELS = {
    "BRAND_NAME",
    "BRAND NAME",
    "PRODUCER_NAME",
    "PRODUCER NAME",
    "CLASS_TYPE",
    "CLASS TYPE",
}


def plausible_brand_value(
    value: str,
) -> bool:
    text = " ".join(
        str(value or "").split()
    ).strip()

    if not text:
        return False

    upper = text.upper()

    if upper in BRAND_FIELD_LABELS:
        return False

    # Reject standalone numbers and confidence scores.
    try:
        float(text)
    except ValueError:
        pass
    else:
        return False

    words = {
        word.strip(".,:;()[]%/").upper()
        for word in text.split()
        if word.strip()
    }

    if words & BRAND_EXCLUDED_TOKENS:
        return False

    if not any(
        character.isalpha()
        for character in text
    ):
        return False

    return True

def resolve_brand_semantically(
    model,
    schema,
    analysis,
    *,
    extraction=None,
):
    unresolved_roles = analysis.get(
        "unresolved_required_roles",
        [],
    )

    if (
        "brand_name"
        not in unresolved_roles
    ):
        return None

    extraction = ensure_extraction(
        model,
        schema,
        analysis,
        extraction,
    )

    if extraction is None:
        return None

    observations = analysis.get(
        "observations",
        [],
    )

    entries = collect_grounded_entities(
        extraction,
        "brand_name",
        observations,
    )

    entries = [
        entry
        for entry in entries
        if plausible_brand_value(
            entry["grounded"].get(
                "value",
                "",
            )
        )
    ]

    # One grounded, plausible brand is safe
    # to resolve.
    if len(entries) == 1:
        return format_grounded_entity(
            entries[0],
            method=(
                "grounded_semantic_"
                "brand_selection"
            ),
        )

    # Multiple plausible grounded brands remain
    # ambiguous. Preserve them for human review.
    if len(entries) > 1:
        analysis[
            "semantic_brand_candidates"
        ] = [
            format_grounded_entity(
                entry,
                method=(
                    "grounded_semantic_"
                    "brand_candidate"
                ),
            )
            for entry in entries
        ]

    return None
