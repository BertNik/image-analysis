from typing import Any

from lib.resolver.semantic.grounding import (
    identity_key
)

def add_semantic_field(
    analysis: dict[str, Any],
    field_name: str,
) -> None:
    fields = analysis.setdefault("semantic_fields", [])
    if field_name not in fields:
        fields.append(field_name)


def apply_semantic_field(
    analysis: dict[str, Any],
    field_name: str,
    value: dict[str, Any],
) -> None:
    analysis.setdefault("resolved", {})[field_name] = value

    unresolved_roles = analysis.get("unresolved_required_roles", [])
    if field_name in unresolved_roles:
        unresolved_roles.remove(field_name)

    analysis.get("unresolved", {}).pop(field_name, None)
    analysis["needs_semantics"] = bool(unresolved_roles)
    analysis["analysis_complete"] = not (
        analysis.get("unresolved_required_fields", [])
        or unresolved_roles
    )
    analysis["deterministic_complete"] = False
    add_semantic_field(analysis, field_name)


def apply_semantic_brand(
    analysis: dict[str, Any],
    semantic_brand: dict[str, Any],
) -> None:
    apply_semantic_field(analysis, "brand_name", semantic_brand)


def apply_semantic_class_type(
    analysis: dict[str, Any],
    semantic_class_type: dict[str, Any],
) -> None:
    apply_semantic_field(analysis, "class_type", semantic_class_type)


def _businesses_by_key(
    businesses: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for business in businesses:
        key = identity_key(business.get("name", ""))
        if key:
            results[key] = dict(business)
    return results


def apply_business_names(
    analysis: dict[str, Any],
    business_names: list[dict[str, Any]],
) -> None:
    resolved = analysis.setdefault("resolved", {})
    businesses = resolved.setdefault("businesses", [])
    businesses_by_key = _businesses_by_key(businesses)

    for semantic_business in business_names:
        name = semantic_business.get("name", "")
        key = identity_key(name)
        if not key:
            continue

        existing = businesses_by_key.get(key, {})
        businesses_by_key[key] = {
            **existing,
            "name": name,
            "observed_text": semantic_business.get("observed_text"),
            "line_indexes": semantic_business.get("line_indexes", []),
            "name_evidence": semantic_business.get("evidence", []),
            "semantic_observations": semantic_business.get(
                "semantic_observations",
                [],
            ),
            "name_grounding_method": semantic_business.get(
                "grounding_method"
            ),
            "name_method": semantic_business.get("method"),
        }

    resolved["businesses"] = list(businesses_by_key.values())


def _append_unique_relation(
    container: list[dict[str, Any]],
    relation: dict[str, Any],
    *,
    identity_fields: tuple[str, ...],
) -> None:
    relation_key = tuple(relation.get(field) for field in identity_fields)
    existing_keys = {
        tuple(item.get(field) for field in identity_fields)
        for item in container
    }
    if relation_key not in existing_keys:
        container.append(relation)


def apply_business_relations(
    analysis: dict[str, Any],
    role_relations: list[dict[str, Any]],
    address_relations: list[dict[str, Any]],
) -> None:
    resolved = analysis.setdefault("resolved", {})
    businesses = resolved.setdefault("businesses", [])
    businesses_by_key = _businesses_by_key(businesses)

    for relation in role_relations:
        key = relation["business_key"]
        business = businesses_by_key.setdefault(
            key,
            {
                "name": relation["business_name"],
                "roles": [],
            },
        )

        roles = set(business.get("roles", []))
        roles.add(relation["role"])
        business["roles"] = sorted(roles)

        evidence = business.setdefault("role_evidence", [])
        _append_unique_relation(
            evidence,
            relation,
            identity_fields=("relation", "business_key"),
        )

    for relation in address_relations:
        key = relation["business_key"]
        business = businesses_by_key.setdefault(
            key,
            {
                "name": relation["business_name"],
                "roles": [],
            },
        )

        existing_address = business.get("address")
        if (
            existing_address
            and identity_key(existing_address)
            != identity_key(relation["address"])
        ):
            conflicts = business.setdefault("address_conflicts", [])
            _append_unique_relation(
                conflicts,
                relation,
                identity_fields=("business_key", "address"),
            )
            continue

        business["address"] = relation["address"]
        business["address_evidence"] = relation["address_evidence"]

        relation_evidence = business.setdefault(
            "address_relation_evidence",
            [],
        )
        _append_unique_relation(
            relation_evidence,
            relation,
            identity_fields=("business_key", "address"),
        )

    resolved["businesses"] = list(businesses_by_key.values())

