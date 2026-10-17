from typing import Any

from .identity import (
    normalize_identity,
)

def reconcile_businesses(
    image_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Combine and deduplicate explicit business relationships.
    """

    businesses_by_key = {}

    for image_result in image_results:
        analysis = image_result.get(
            "analysis",
            {},
        )

        businesses = (
            analysis
            .get("resolved", {})
            .get("businesses", [])
        )

        for business in businesses:
            name = (
                business.get("name")
                or business.get("value")
                or ""
            )

            key = normalize_identity(name)

            if not key:
                continue

            source = {
                **business,
                "image_index":
                    image_result.get(
                        "image_index"
                    ),
                "filename":
                    image_result.get(
                        "filename"
                    ),
            }

            if key not in businesses_by_key:
                businesses_by_key[key] = {
                    **business,
                    "evidence": [source],
                }

                continue

            businesses_by_key[
                key
            ]["evidence"].append(
                source
            )

            existing_roles = set(
                businesses_by_key[
                    key
                ].get(
                    "roles",
                    [],
                )
            )

            existing_roles.update(
                business.get(
                    "roles",
                    [],
                )
            )

            businesses_by_key[
                key
            ]["roles"] = sorted(
                existing_roles
            )

    return list(
        businesses_by_key.values()
    )