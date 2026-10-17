from __future__ import annotations

from typing import Any

from .brand import (
    reconcile_brand,
)
from .business import (
    reconcile_businesses,
)
from .candidates import (
    collect_case_candidates,
)
from .constants import (
    OPTIONAL_CASE_FIELDS,
    REQUIRED_CASE_FIELDS,
)
from .field import (
    reconcile_field,
)
from .warning import (
    government_warning_reasons,
)


def reconcile_case(
    image_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Merge multiple image-level analyses into one label-review case.

    Rules:
      - Evidence may come from any submitted image.
      - Matching values are merged.
      - Conflicting values are never resolved by confidence alone.
      - Required values missing everywhere route to VERIFY.
      - A case with no usable image results routes to ABSTAIN.
    """

    if not image_results:
        return {
            "status": "ABSTAIN",
            "reasons": [
                "no_images_processed"
            ],
            "images": [],
            "analysis": {
                "resolved": {},
                "conflicts": {},
                "unresolved_required_fields":
                    list(
                        REQUIRED_CASE_FIELDS
                    ),
                "not_observed_fields":
                    list(
                        OPTIONAL_CASE_FIELDS
                    ),
                "analysis_complete": False,
            },
        }

    usable_results = [
        result
        for result in image_results
        if (
            result.get("status")
            != "ERROR"
            and result.get(
                "usability",
                {},
            ).get(
                "usable",
                False,
            )
        )
    ]

    if not usable_results:
        return {
            "status": "ABSTAIN",
            "reasons": [
                "no_usable_image_evidence"
            ],
            "images": [
                {
                    "image_index":
                        result.get(
                            "image_index"
                        ),

                    "filename":
                        result.get(
                            "filename"
                        ),

                    "status":
                        result.get(
                            "status"
                        ),

                    "error":
                        result.get(
                            "error"
                        ),
                }
                for result in image_results
            ],
            "analysis": {
                "resolved": {},
                "conflicts": {},
                "unresolved_required_fields":
                    list(
                        REQUIRED_CASE_FIELDS
                    ),
                "not_observed_fields":
                    list(
                        OPTIONAL_CASE_FIELDS
                    ),
                "analysis_complete": False,
            },
        }

    reconciled_fields = {}

    for field_name in (
        *REQUIRED_CASE_FIELDS,
        *OPTIONAL_CASE_FIELDS,
    ):
        if field_name == "brand_name":
            field_result = reconcile_brand(
                usable_results
            )
        else:
            field_result = reconcile_field(
                field_name,
                usable_results,
            )

        if field_result is None:
            raise RuntimeError(
                "Reconciliation returned None "
                f"for field: {field_name}"
            )

        reconciled_fields[
            field_name
        ] = field_result

    resolved = {}
    conflicts = {}
    unresolved_required_fields = []
    not_observed_fields = []

    for field_name in REQUIRED_CASE_FIELDS:
        field_result = reconciled_fields[
            field_name
        ]

        if (
            field_result["status"]
            == "RESOLVED"
        ):
            resolved[field_name] = (
                field_result["value"]
            )

        elif (
            field_result["status"]
            == "CONFLICT"
        ):
            resolved[field_name] = None

            conflicts[field_name] = (
                field_result["conflicts"]
            )

        else:
            resolved[field_name] = None

            unresolved_required_fields.append(
                field_name
            )

    for field_name in OPTIONAL_CASE_FIELDS:
        field_result = reconciled_fields[
            field_name
        ]

        if (
            field_result["status"]
            == "RESOLVED"
        ):
            resolved[field_name] = (
                field_result["value"]
            )

        elif (
            field_result["status"]
            == "CONFLICT"
        ):
            resolved[field_name] = None

            conflicts[field_name] = (
                field_result["conflicts"]
            )

        else:
            resolved[field_name] = None

            not_observed_fields.append(
                field_name
            )

    resolved["businesses"] = (
        reconcile_businesses(
            usable_results
        )
    )
        
    reasons = []

    for field_name in (
        unresolved_required_fields
    ):
        reasons.append(
            f"{field_name}_not_resolved"
        )

    for field_name in conflicts:
        reasons.append(
            f"{field_name}_conflict"
        )

    for reason in (
        government_warning_reasons(
            usable_results
        )
    ):
        if reason not in reasons:
            reasons.append(
                reason
            )

    if reasons:
        case_status = "VERIFY"
    else:
        case_status = "RESOLVED"

    # Preserve whether any image used semantic assistance.
    semantic_fields = sorted({
        field
        for result in usable_results
        for field in (
            result
            .get("analysis", {})
            .get(
                "semantic_fields",
                [],
            )
        )
    })

    image_summaries = []

    for result in image_results:
        image_summaries.append(
            {
                "image_index":
                    result.get(
                        "image_index"
                    ),

                "filename":
                    result.get(
                        "filename"
                    ),

                "status":
                    result.get(
                        "status"
                    ),

                "reasons":
                    result.get(
                        "reasons",
                        [],
                    ),

                "timing":
                    result.get(
                        "timing",
                        {},
                    ),

                # Keep the image-level analysis available for
                # audit and image switching in the UI.
                "analysis":
                    result.get(
                        "analysis"
                    ),

                "error":
                    result.get(
                        "error"
                    ),
            }
        )

    total_seconds = sum(
        float(
            result
            .get("timing", {})
            .get(
                "total_seconds",
                0.0,
            )
            or 0.0
        )
        for result in image_results
    )
    class_type_candidates = (
        collect_case_candidates(
            usable_results,
            "class_type",
        )
    )

    country_of_origin_candidates = (
            collect_case_candidates(
                usable_results,
                "country_of_origin",
            )
        )


    return {
        "status":
            case_status,

        "reasons":
            reasons,

        "timing": {
            "total_seconds":
                total_seconds,

            "image_count":
                len(image_results),

            "usable_image_count":
                len(usable_results),
        },

        "analysis": {
            "resolved":
                resolved,

            "conflicts":
                conflicts,

            "unresolved_required_fields":
                unresolved_required_fields,

            "not_observed_fields":
                not_observed_fields,

            "semantic_fields":
                semantic_fields,

            "semantic_class_type_candidates":
                class_type_candidates,

            "semantic_country_of_origin_candidates":
                country_of_origin_candidates,
            
            "analysis_complete":
                not reasons,

            "image_count":
                len(image_results),

            "usable_image_count":
                len(usable_results),
        },

        "images":
            image_summaries,

        "error":
            None,
    }