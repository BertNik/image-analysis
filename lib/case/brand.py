from typing import Any

from .field import (
    add_image_provenance,
    evidence_confidence,
)

def reconcile_brand(
    image_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Reconcile brand evidence across one or more images.

    For one image, preserve a confidently resolved image-level
    brand. For multiple images, use cross-image consensus.
    """

    if not image_results:
        return {
            "status": "MISSING",
            "value": None,
            "evidence": [],
            "conflicts": [],
        }

    # --------------------------------------------------------
    # Single-image case
    #
    # No cross-image consensus is possible or necessary.
    # Preserve the image-level deterministic result.
    # --------------------------------------------------------
    for image_result in image_results:
       
        resolved_brand = (
            image_result
            .get("analysis", {})
            .get("resolved", {})
            .get("brand_name")
        )

        if resolved_brand:
            evidence = add_image_provenance(
                resolved_brand,
                image_result,
            )

            reconciled = dict(
                evidence
            )

            reconciled["evidence"] = [
                {
                    "image_index":
                        evidence.get(
                            "image_index"
                        ),

                    "filename":
                        evidence.get(
                            "filename"
                        ),

                    "line_indexes":
                        evidence.get(
                            "line_indexes",
                            [],
                        ),

                    "confidence":
                        evidence_confidence(
                            evidence
                        ),

                    "method":
                        evidence.get(
                            "method"
                        ),

                    "value":
                        evidence.get(
                            "value"
                        ),
                }
            ]

            reconciled[
                "direct_source_count"
            ] = 1

            reconciled[
                "source_count"
            ] = 1

            reconciled[
                "reconciliation_method"
            ] = "single_image_resolved_brand"

            return {
                "status": "RESOLVED",
                "value": reconciled,
                "evidence": [evidence],
                "conflicts": [],
            }

        # The single-image analyzer did not resolve a brand.
        return {
            "status": "MISSING",
            "value": None,
            "evidence": [],
            "conflicts": [],
        }