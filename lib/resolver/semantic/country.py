from __future__ import annotations

from lib.resolver.semantic.extraction import ensure_extraction
from lib.resolver.semantic.grounding import (
    collect_grounded_entities, 
    format_grounded_entity, 
    get_entity_values
)

def resolve_country_of_origin_semantically(
    model,
    schema,
    analysis,
    *,
    extraction=None,
):
    
    extraction = ensure_extraction(
        model,
        schema,
        analysis,
        extraction,
    )

    if not isinstance(
        extraction,
        dict,
    ):
        return None

    observations = analysis.get(
        "observations",
        [],
    )

    entries = collect_grounded_entities(
        extraction,
        "country_of_origin",
        observations,
    )

    if len(entries) == 1:
        return format_grounded_entity(
            entries[0],
            method=(
                "grounded_semantic_"
                "country_of_origin_selection"
            ),
        )

    if len(entries) > 1:
        analysis[
            "semantic_country_of_origin_candidates"
        ] = [
            format_grounded_entity(
                entry,
                method=(
                    "grounded_semantic_"
                    "country_of_origin_candidate"
                ),
            )
            for entry in entries
        ]

        return None

    # No uniquely grounded entry was available,
    # but preserve the model's raw values for review.
    raw_candidates = get_entity_values(
        extraction,
        "country_of_origin",
    )

    if raw_candidates:
        analysis[
            "semantic_country_of_origin_candidates"
        ] = [
            {
                "value": value,
                "semantic_observations": [
                    value,
                ],
                "method": (
                    "semantic_country_of_origin_candidate"
                ),
            }
            for value in raw_candidates
        ]

    return None



