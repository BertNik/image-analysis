from typing import Any
from collections import defaultdict

from .identity import canonical_field_key

def reconcile_contained_text_groups(
    groups: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    """
    Resolve compatible textual variants when shorter values are
    contained within a more specific longer value.

    Example:
        BOURBON WHISKEY
        KENTUCKY STRAIGHT BOURBON WHISKEY

    Result:
        KENTUCKY STRAIGHT BOURBON WHISKEY
    """

    if not groups:
        return None

    if len(groups) == 1:
        key, evidence = next(
            iter(groups.items())
        )

        return {
            "key": key,
            "evidence": evidence,
        }

    ranked = sorted(
        groups.items(),
        key=lambda item: (
            len(item[0]),
            len(item[1]),
        ),
        reverse=True,
    )

    longest_key, longest_evidence = (
        ranked[0]
    )

    for other_key, _ in ranked[1:]:
        if other_key not in longest_key:
            return None

    return {
        "key": longest_key,
        "evidence": longest_evidence,
    }

def evidence_confidence(
    field: dict[str, Any],
) -> float:
    """
    Retrieve the best available confidence-like value.

    This is used only to choose the representative observation
    when multiple images agree. It is never used to override
    conflicting field values.
    """

    for key in (
        "ocr_confidence",
        "confidence",
    ):
        value = field.get(key)

        if value is None:
            continue

        try:
            return float(value)
        except (TypeError, ValueError):
            pass

    return 0.0


def add_image_provenance(
    field: dict[str, Any],
    image_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Copy a field and attach the source-image information.
    """

    evidence = dict(field)

    evidence["image_index"] = (
        image_result.get("image_index")
    )

    evidence["filename"] = (
        image_result.get("filename")
    )

    return evidence


def reconcile_field(
    field_name: str,
    image_results: list[dict[str, Any]],
) -> dict[str, Any]:

    observations = []

    for image_result in image_results:
        analysis = image_result.get(
            "analysis",
            {},
        )

        resolved = analysis.get(
            "resolved",
            {},
        )

        field = resolved.get(
            field_name
        )

        if not field:
            continue

        key = canonical_field_key(
            field_name,
            field,
        )

        if key is None:
            continue

        observations.append(
            {
                "key": key,
                "field":
                    add_image_provenance(
                        field,
                        image_result,
                    ),
            }
        )

    # Nothing observed
    if not observations:
        return {
            "status": "MISSING",
            "value": None,
            "evidence": [],
            "conflicts": [],
        }

    groups = defaultdict(list)

    for observation in observations:
        groups[
            observation["key"]
        ].append(
            observation["field"]
        )

    # --------------------------------------------------------
    # Multiple different canonical values
    # --------------------------------------------------------

    if len(groups) > 1:

        compatible = None

        if field_name in {
            "brand_name",
            "class_type",
        }:
            compatible = (
                reconcile_contained_text_groups(
                    groups
                )
            )

        # Compatible partial readings
        if compatible is not None:

            selected_key = (
                compatible["key"]
            )

            selected_evidence = (
                compatible["evidence"]
            )

            representative = max(
                selected_evidence,
                key=evidence_confidence,
            )

            reconciled = dict(
                representative
            )

            all_compatible_evidence = [
                item

                for key, items
                in groups.items()

                if key in selected_key

                for item in items
            ]

            reconciled["evidence"] = [
                {
                    "image_index":
                        item.get(
                            "image_index"
                        ),

                    "filename":
                        item.get(
                            "filename"
                        ),

                    "line_indexes":
                        item.get(
                            "line_indexes",
                            [],
                        ),

                    "confidence":
                        evidence_confidence(
                            item
                        ),

                    "method":
                        item.get(
                            "method"
                        ),

                    "observed_text":
                        item.get(
                            "observed_text"
                        ),

                    "value":
                        item.get(
                            "value"
                        ),
                }

                for item
                in all_compatible_evidence
            ]

            reconciled[
                "source_count"
            ] = len(
                all_compatible_evidence
            )

            reconciled[
                "reconciliation_method"
            ] = (
                "most_specific_compatible_value"
            )

            return {
                "status":
                    "RESOLVED",

                "value":
                    reconciled,

                "evidence":
                    all_compatible_evidence,

                "conflicts":
                    [],
            }

        # Genuine conflict
        return {
            "status": "CONFLICT",
            "value": None,

            "evidence": [
                item
                for items
                in groups.values()
                for item
                in items
            ],

            "conflicts": [
                {
                    "canonical_key":
                        key,

                    "evidence":
                        evidence,
                }

                for key, evidence
                in groups.items()
            ],
        }

    # --------------------------------------------------------
    # Single canonical value
    # --------------------------------------------------------

    evidence = next(
        iter(
            groups.values()
        )
    )

    representative = max(
        evidence,
        key=evidence_confidence,
    )

    reconciled = dict(
        representative
    )

    reconciled["evidence"] = [
        {
            "image_index":
                item.get(
                    "image_index"
                ),

            "filename":
                item.get(
                    "filename"
                ),

            "line_indexes":
                item.get(
                    "line_indexes",
                    [],
                ),

            "confidence":
                evidence_confidence(
                    item
                ),

            "method":
                item.get(
                    "method"
                ),

            "observed_text":
                item.get(
                    "observed_text"
                ),
        }

        for item in evidence
    ]

    reconciled[
        "source_count"
    ] = len(
        evidence
    )

    return {
        "status":
            "RESOLVED",

        "value":
            reconciled,

        "evidence":
            evidence,

        "conflicts":
            [],
    }