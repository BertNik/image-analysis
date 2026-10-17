from typing import Any

from lib.resolver.semantic.grounding import (
    collect_grounded_entities, 
    format_grounded_entities, 
    ground_semantic_span,
    identity_key
)
from lib.resolver.semantic.extraction import (
    ensure_extraction,
    get_relation_pairs,
)
from lib.resolver.semantic.schema import BUSINESS_ROLE_BY_RELATION

def resolve_business_names_semantically(
    model,
    schema,
    analysis: dict[str, Any],
    *,
    extraction: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    extraction = ensure_extraction(
        model,
        schema,
        analysis,
        extraction,
    )
    if extraction is None:
        return []

    entries = collect_grounded_entities(
        extraction,
        "business_name",
        analysis.get("observations", []),
    )
    return format_grounded_entities(
        entries,
        value_key="name",
        method="grounded_semantic_business_name",
    )


def resolve_business_addresses_semantically(
    model,
    schema,
    analysis: dict[str, Any],
    *,
    extraction: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    extraction = ensure_extraction(
        model,
        schema,
        analysis,
        extraction,
    )
    if extraction is None:
        return []

    entries = collect_grounded_entities(
        extraction,
        "business_address",
        analysis.get("observations", []),
    )
    return format_grounded_entities(
        entries,
        method="grounded_semantic_business_address",
    )


def resolve_business_role_relations(
    extraction: dict[str, Any],
    observations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Ground the business target for every language-independent role fact."""
    results: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for relation_name, canonical_role in BUSINESS_ROLE_BY_RELATION.items():
        for source, target in get_relation_pairs(extraction, relation_name):
            grounded_business = ground_semantic_span(target, observations)
            if grounded_business is None:
                continue

            business_key = identity_key(grounded_business.get("value", ""))
            if not business_key:
                continue

            result_key = (relation_name, business_key)
            if result_key in seen:
                continue
            seen.add(result_key)

            results.append(
                {
                    "relation": relation_name,
                    "role": canonical_role,
                    "source": source,
                    "business_name": grounded_business["value"],
                    "business_key": business_key,
                    "business_evidence": grounded_business.get("evidence", []),
                    "business_line_indexes": grounded_business.get(
                        "line_indexes",
                        [],
                    ),
                    "grounding_method": grounded_business.get("method"),
                    "method": "grounded_semantic_business_relation",
                }
            )

    return results


def resolve_business_address_relations(
    extraction: dict[str, Any],
    observations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for business, address in get_relation_pairs(
        extraction,
        "business_located_at",
    ):
        grounded_business = ground_semantic_span(business, observations)
        grounded_address = ground_semantic_span(address, observations)
        if grounded_business is None or grounded_address is None:
            continue

        business_key = identity_key(grounded_business.get("value", ""))
        address_key = identity_key(grounded_address.get("value", ""))
        if not business_key or not address_key:
            continue

        result_key = (business_key, address_key)
        if result_key in seen:
            continue
        seen.add(result_key)

        results.append(
            {
                "relation": "business_located_at",
                "business_name": grounded_business["value"],
                "business_key": business_key,
                "address": grounded_address["value"],
                "business_evidence": grounded_business.get("evidence", []),
                "address_evidence": grounded_address.get("evidence", []),
                "method": "grounded_semantic_business_address_relation",
            }
        )

    return results

