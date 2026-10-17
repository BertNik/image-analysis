from __future__ import annotations

from typing import Any

from ...label.grounding import (
    ground_semantic_span,
    identity_key,
)
from .extraction import (
    get_entity_values,
)

def collect_grounded_entities(
    extraction: dict[str, Any],
    label: str,
    observations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extract, ground, and deduplicate entity values by normalized identity."""
    grounded_by_key: dict[str, dict[str, Any]] = {}
    
    for extracted_value in get_entity_values(extraction, label):
        
        grounded = ground_semantic_span(extracted_value, observations)
        if grounded is None:
            continue
        # print(grounded)

        key = identity_key(grounded.get("value", ""))
        if not key:
            continue

        entry = grounded_by_key.setdefault(
            key,
            {
                "grounded": grounded,
                "semantic_observations": [],
            },
        )

        semantic_observations = entry["semantic_observations"]
        if extracted_value not in semantic_observations:
            semantic_observations.append(extracted_value)

    return list(grounded_by_key.values())


def format_grounded_entity(
    entry: dict[str, Any],
    *,
    method: str,
    value_key: str = "value",
) -> dict[str, Any]:
    grounded = entry["grounded"]
    return {
        value_key: grounded["value"],
        "observed_text": grounded.get("observed_text"),
        "line_indexes": grounded.get("line_indexes", []),
        "evidence": grounded.get("evidence", []),
        "semantic_observations": entry["semantic_observations"],
        "grounding_method": grounded.get("method"),
        "method": method,
    }


def format_grounded_entities(
    entries: list[dict[str, Any]],
    *,
    method: str,
    value_key: str = "value",
) -> list[dict[str, Any]]:
    return [
        format_grounded_entity(
            entry,
            method=method,
            value_key=value_key,
        )
        for entry in entries
    ]
