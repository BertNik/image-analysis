from __future__ import annotations

from typing import Any
from lib.resolver.semantic.extraction import (
    ensure_extraction,
)
from lib.resolver.semantic.brand import (
    resolve_brand_semantically,
)
from lib.resolver.semantic.apply import (
    apply_semantic_field,
    apply_business_names,
    apply_business_relations
)
from lib.resolver.class_type import resolve_class_type_semantically
from lib.resolver.semantic.business import resolve_business_names_semantically
from lib.resolver.semantic.apply import add_semantic_field
from lib.resolver.semantic.business import  (
    resolve_business_addresses_semantically, 
    resolve_business_address_relations, 
    resolve_business_role_relations
)


def resolve_semantic_roles(
    model,
    schema,
    analysis: dict[str, Any],
) -> dict[str, Any]:
    """
    Run one semantic inference and orchestrate
    resolution and application.
    """
    results: dict[str, Any] = {}

    observations = analysis.get(
        "observations",
        [],
    )

    ocr_text = str(
        analysis.get(
            "ocr_text",
            "",
        )
        or ""
    ).strip()

    if (
        not observations
        or not ocr_text
    ):
        return results, None

    # Run semantic inference once.
    extraction = ensure_extraction(
        model,
        schema,
        analysis,
        extraction=None,
    )

    if not isinstance(
        extraction,
        dict,
    ):
        return results, None

    unresolved_roles = analysis.get(
        "unresolved_required_roles",
        [],
    )

    # ====================================================
    # Brand name
    # ====================================================

    if (
        "brand_name"
        in unresolved_roles
    ):
        brand = resolve_brand_semantically(
            model,
            schema,
            analysis,
            extraction=extraction,
        )

        if brand is not None:
            apply_semantic_field(
                analysis,
                "brand_name",
                brand,
            )

            results[
                "brand_name"
            ] = brand

    # apply_semantic_field mutates this list,
    # so refresh the reference.
    unresolved_roles = analysis.get(
        "unresolved_required_roles",
        [],
    )

    # ====================================================
    # Class/type
    # ====================================================

    if (
        "class_type"
        in unresolved_roles
    ):
        class_type = (
            resolve_class_type_semantically(
                model,
                schema,
                analysis,
                extraction=extraction,
            )
        )

        if class_type is not None:
            apply_semantic_field(
                analysis,
                "class_type",
                class_type,
            )

            results[
                "class_type"
            ] = class_type

    # ====================================================
    # Business names
    # ====================================================

    business_names = (
        resolve_business_names_semantically(
            model,
            schema,
            analysis,
            extraction=extraction,
        )
    )

    if business_names:
        apply_business_names(
            analysis,
            business_names,
        )

        results[
            "business_names"
        ] = business_names

        add_semantic_field(
            analysis,
            "business_name",
        )

    # ====================================================
    # Business addresses
    # ====================================================

    business_addresses = (
        resolve_business_addresses_semantically(
            model,
            schema,
            analysis,
            extraction=extraction,
        )
    )

    if business_addresses:
        analysis[
            "semantic_business_addresses"
        ] = business_addresses

        results[
            "business_addresses"
        ] = business_addresses

        add_semantic_field(
            analysis,
            "business_address",
        )

    # ====================================================
    # Business relationships
    # ====================================================

    business_role_relations = (
        resolve_business_role_relations(
            extraction,
            observations,
        )
    )

    business_address_relations = (
        resolve_business_address_relations(
            extraction,
            observations,
        )
    )

    if (
        business_role_relations
        or business_address_relations
    ):
        apply_business_relations(
            analysis,
            business_role_relations,
            business_address_relations,
        )

    if business_role_relations:
        results[
            "business_role_relations"
        ] = business_role_relations

        add_semantic_field(
            analysis,
            "business_role",
        )

    if business_address_relations:
        results[
            "business_address_relations"
        ] = business_address_relations

        add_semantic_field(
            analysis,
            "business_address_relation",
        )

    # ====================================================
    # Return unresolved candidate lists
    # ====================================================

    for result_key, analysis_key in (
        (
            "brand_candidates",
            "semantic_brand_candidates",
        ),
        (
            "class_type_candidates",
            "semantic_class_type_candidates",
        ),
    ):
        candidates = analysis.get(
            analysis_key,
            [],
        )

        if candidates:
            results[
                result_key
            ] = candidates

    return results, extraction