from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np

from .label.analyzer import (
    analyze_label_deterministically,
)
from .ocr import ocr_function
from .routing import route_analysis
from .resolver.semantic.country import (
    resolve_country_of_origin_semantically,
)
from lib.resolver.semantic.orchestrator import resolve_semantic_roles

from lib.warning_label import (
    validate_government_warning,
)
from lib.resolver.visual import (
    analyze_heading_boldness
)

def analyze_image(
    image: np.ndarray,
    *,
    semantic_model=None,
    semantic_schema=None,
    brand_required: bool = False,
) -> dict[str, Any]:
    """
    Run the canonical in-memory label-analysis pipeline.

    Processing flow:
        decoded image
        -> OCR
        -> deterministic extraction
        -> initial routing
        -> grounded semantic resolution
        -> final routing
    """

    total_start = perf_counter()

    # ========================================================
    # OCR
    # ========================================================

    ocr_start = perf_counter()

    ocr_result = ocr_function(
        image
    )

    ocr_seconds = (
        perf_counter()
        - ocr_start
    )

    # ========================================================
    # Deterministic analysis
    # ========================================================

    deterministic_start = (
        perf_counter()
    )

    analysis = (
        analyze_label_deterministically(
            ocr_result,
            brand_required=brand_required,
        )
    )

    deterministic_seconds = (
        perf_counter()
        - deterministic_start
    )

    # ========================================================
    # Warning visual analysis
    # ========================================================

    warning = analysis.get(
        "government_warning"
    )

    if warning is not None:
        heading_box = warning.get(
            "heading_box"
        )

        if heading_box is None:
            for evidence in warning.get(
                "evidence",
                [],
            ):
                if (
                    str(
                        evidence.get(
                            "text",
                            ""
                        )
                    ).strip()
                    == "GOVERNMENT WARNING:"
                ):
                    heading_box = (
                        evidence.get(
                            "box"
                        )
                    )
                    break

        if heading_box is not None:
            boldness = (
                analyze_heading_boldness(
                    image=image,
                    heading_box=heading_box,
                    observations=analysis.get(
                        "observations",
                        [],
                    ),
                    warning_line_indexes=(
                        warning.get(
                            "line_indexes",
                            [],
                        )
                    ),
                )
            )
        else:
            boldness = {
                "value": None,
                "reason": (
                    "warning_heading_box_"
                    "unavailable"
                ),
                "method": (
                    "relative_stroke_width"
                ),
            }

        warning["heading_bold"] = (
            boldness.get("value")
        )

        warning[
            "boldness_evidence"
        ] = boldness

        analysis[
            "government_warning"
        ] = (
            validate_government_warning(
                warning
            )
        )

    deterministic_seconds = (
        perf_counter()
        - deterministic_start
    )

    # ========================================================
    # Initial routing
    # ========================================================

    routing = route_analysis(
        analysis
    )

    semantic_result = None
    semantic_seconds = 0.0

    # ========================================================
    # Semantic resolution
    # ========================================================

    # Semantic execution is driven by unresolved semantic
    # roles, not by the current routing status.
    #
    # A missing deterministic field may already produce VERIFY,
    # but any available brand/class-type evidence should still
    # be semantically resolved and preserved.
    semantic_result_country = []
    if analysis.get(
        "needs_semantics",
        False,
    ):  
        if (
            semantic_model is not None
            and semantic_schema is not None
        ):
            semantic_start = (
                perf_counter()
            )

            semantic_result, extraction = (
                resolve_semantic_roles(
                    semantic_model,
                    semantic_schema,
                    analysis,
                )
            )
            semantic_result_country = (
                resolve_country_of_origin_semantically(
                    semantic_model,
                    semantic_schema,
                    analysis,
                    extraction=extraction
                )
            )

            semantic_seconds = (
                perf_counter()
                - semantic_start
            )

            # Applying semantic results mutates analysis.
            # Route the updated state again.
            routing = route_analysis(
                analysis
            )

            # Semantic processing ran, but one or more
            # required semantic roles remain unresolved.
            if (
                routing["status"]
                == "SEMANTIC_FALLBACK"
            ):
                unresolved_roles = (
                    analysis.get(
                        "unresolved_required_roles",
                        [],
                    )
                )

                routing = {
                    "status": "VERIFY",

                    "reasons": [
                        (
                            f"{role}_"
                            "semantic_unresolved"
                        )
                        for role
                        in unresolved_roles
                    ],

                    "usability":
                        routing["usability"],
                }

        elif (
            routing["status"]
            == "SEMANTIC_FALLBACK"
        ):
            # Semantic roles are the only current blockers,
            # but the semantic dependencies are unavailable.
            routing = {
                "status": "VERIFY",

                "reasons": [
                    "semantic_model_unavailable",
                ],

                "usability":
                    routing["usability"],
            }

    # ========================================================
    # Final result
    # ========================================================

    return {
        "usability":
            routing["usability"],

        "status":
            routing["status"],

        "reasons":
            routing["reasons"],

        "timing": {
            "ocr_seconds":
                ocr_seconds,

            "deterministic_seconds":
                deterministic_seconds,

            "semantic_seconds":
                semantic_seconds,

            "total_seconds": (
                perf_counter()
                - total_start
            ),
        },

        # Mapping of each semantic field resolved during this
        # attempt, or None if semantics were not attempted.
        "semantic_result":
            semantic_result,

        "semantic_result_country":
            semantic_result_country,

        "analysis":
            analysis,

        "error":
            None,
    }