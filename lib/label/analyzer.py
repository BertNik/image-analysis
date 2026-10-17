import re
from typing import Any, Optional

from .measurements import (
    ABV_PATTERN,
    PROOF_PATTERN,
    PROOF_LIKE_PATTERN,
    NET_CONTENTS_PATTERN,
    extract_measurements,
    resolve_proof_constraint,
)

from ..warning_label import(
    extract_government_warning,
    validate_government_warning,
)

def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()

def build_observations(
    ocr_result: dict[str, Any],
) -> list[dict[str, Any]]:
    lines = ocr_result.get("lines", [])
    scores = ocr_result.get("scores", [])
    boxes = ocr_result.get("boxes", [])

    observations = []

    for index, text in enumerate(lines):
        observations.append(
            {
                "index": index,
                "text": normalize_space(text),
                "confidence": (
                    float(scores[index])
                    if index < len(scores)
                    else None
                ),
                "box": (
                    boxes[index]
                    if index < len(boxes)
                    else None
                ),
            }
        )

    return observations

def extract_measurements(
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    measurements = {
        "alcohol_content": None,
        "proof": None,
        "net_contents": None,
        "unresolved_measurements": [],
    }

    for item in observations:
        text = item["text"]
    
        if measurements["alcohol_content"] is None:
            match = ABV_PATTERN.search(text)

            if match:
                measurements["alcohol_content"] = {
                    "value": float(
                        match.group("value")
                    ),
                    "unit": "percent_abv",
                    "observed_text": text,
                    "line_indexes": [
                        item["index"]
                    ],
                    "confidence": item["confidence"],
                    "method": "regex",
                }

        if measurements["net_contents"] is None:
            match = NET_CONTENTS_PATTERN.search(text)

            if match:
                raw_unit = (
                    match.group("unit").upper()
                )

                unit = (
                    "mL"
                    if raw_unit.startswith("M")
                    else "L"
                )

                measurements["net_contents"] = {
                    "value": float(
                        match.group("value")
                    ),
                    "unit": unit,
                    "observed_text": text,
                    "line_indexes": [
                        item["index"]
                    ],
                    "confidence": item["confidence"],
                    "method": "regex",
                }

        if measurements["proof"] is None:
            match = PROOF_PATTERN.search(text)

            if match:
                measurements["proof"] = {
                    "value": float(
                        match.group("value")
                    ),
                    "unit": "proof",
                    "observed_text": text,
                    "line_indexes": [
                        item["index"]
                    ],
                    "confidence": item["confidence"],
                    "method": "regex",
                }

            else:
                proof_like_match = (
                    PROOF_LIKE_PATTERN.search(text)
                )

                if proof_like_match:
                    measurements[
                        "unresolved_measurements"
                    ].append(
                        {
                            "type": "proof_like",
                            "candidate_value": float(
                                proof_like_match.group(
                                    "value"
                                )
                            ),
                            "observed_text": text,
                            "line_indexes": [
                                item["index"]
                            ],
                            "confidence": (
                                item["confidence"]
                            ),
                        }
                    )

    return measurements


def resolve_proof_constraint(
    measurements: dict[str, Any],
    tolerance: float = 0.01,
) -> Optional[dict[str, Any]]:
    if measurements["proof"] is not None:
        return measurements["proof"]

    alcohol = measurements["alcohol_content"]

    if alcohol is None:
        return None

    expected_proof = alcohol["value"] * 2

    for candidate in measurements[
        "unresolved_measurements"
    ]:
        if candidate["type"] != "proof_like":
            continue

        candidate_value = candidate[
            "candidate_value"
        ]

        if (
            abs(
                candidate_value
                - expected_proof
            )
            <= tolerance
        ):
            return {
                "value": candidate_value,
                "unit": "proof",
                "observed_text": (
                    candidate["observed_text"]
                ),
                "line_indexes": (
                    candidate["line_indexes"]
                ),
                "confidence": (
                    candidate["confidence"]
                ),
                "method": (
                    "cross_field_constraint"
                ),
                "evidence": {
                    "abv": alcohol["value"],
                    "expected_proof": (
                        expected_proof
                    ),
                },
            }

    return None

def analyze_label_deterministically(
    ocr_result: dict[str, Any],
    brand_required: bool = True,
) -> dict[str, Any]:
    """
    Resolve fields that can be established deterministically.

    Semantic roles such as brand name and class/type are left
    unresolved for the grounded semantic resolver.

    Deterministic fields currently include:
        - alcohol content
        - proof
        - net contents
        - explicit business relationships
    """

    # ========================================================
    # Build structured OCR observations
    # ========================================================

    observations = build_observations(
        ocr_result
    )

    # ========================================================
    # Resolve deterministic fields
    # ========================================================

    measurements = extract_measurements(
        observations
    )
    
    resolved_proof = (
        resolve_proof_constraint(
            measurements
        )
    )

    businesses = []

    # ========================================================
    # Semantic roles
    # ========================================================

    # Brand and class/type require semantic role assignment.
    #
    # The deterministic analyzer must not infer either role
    # through position, prominence, vocabulary ranking,
    # confidence ranking, or other heuristic scoring.

    brand = None
    class_type = None

    # ========================================================
    # Classify unresolved requirements
    # ========================================================

    unresolved_required_fields = []
    unresolved_required_roles = []
    unresolved = {}

    # --------------------------------------------------------
    # Government Health Warning
    # --------------------------------------------------------

    warning_candidate = (
        extract_government_warning(
            observations
        )
    )

    government_warning = (
        validate_government_warning(
            warning_candidate
        )
    )


    # --------------------------------------------------------
    # Alcohol content
    # --------------------------------------------------------

    if (
        measurements.get(
            "alcohol_content"
        )
        is None
    ):
        unresolved_required_fields.append(
            "alcohol_content"
        )

        unresolved[
            "alcohol_content"
        ] = {
            "reason":
                "alcohol_content_not_observed"
        }

    # --------------------------------------------------------
    # Class/type
    # --------------------------------------------------------

    # Class/type is now a semantic role rather than a
    # deterministic vocabulary match.

    unresolved_required_roles.append(
        "class_type"
    )

    unresolved["class_type"] = {
        "reason":
            "class_type_requires_semantic_resolution"
    }

   
        
    # --------------------------------------------------------
    # Brand
    # --------------------------------------------------------

    if brand_required:
        unresolved_required_roles.append(
            "brand_name"
        )

        unresolved["brand_name"] = {
            "reason":
                "brand_requires_semantic_resolution"
        }

    # ========================================================
    # Track optional fields not observed in this image
    # ========================================================

    not_observed_fields = []

    # Net contents may appear elsewhere on the container.
    if (
        measurements.get(
            "net_contents"
        )
        is None
    ):
        not_observed_fields.append(
            "net_contents"
        )

    # ========================================================
    # Completion / routing state
    # ========================================================

    needs_semantics = bool(
        unresolved_required_roles
    )

    deterministic_complete = not (
        unresolved_required_fields
        or unresolved_required_roles
    )

    analysis_complete = (
        deterministic_complete
    )

    # ========================================================
    # Return analysis
    # ========================================================

    return {
        "ocr_text":
            ocr_result.get(
                "text",
                "",
            ),

        "government_warning":
            government_warning,

        "observations":
            observations,

        "resolved": {
            "brand_name":
                brand,

            "class_type":
                class_type,

            "alcohol_content":
                measurements.get(
                    "alcohol_content"
                ),

            "proof":
                resolved_proof,

            "net_contents":
                measurements.get(
                    "net_contents"
                ),

            "businesses":
                businesses,
        },

        "unresolved_measurements":
            measurements.get(
                "unresolved_measurements",
                [],
            ),

        "unresolved":
            unresolved,

        "unresolved_required_fields":
            unresolved_required_fields,

        "unresolved_required_roles":
            unresolved_required_roles,

        "not_observed_fields":
            not_observed_fields,

        "semantic_fields":
            [],

        "deterministic_complete":
            deterministic_complete,

        "analysis_complete":
            analysis_complete,

        "needs_semantics":
            needs_semantics,
    }

def resolve_label(
    ocr_result: dict[str, Any],
    brand_required: bool = False,
) -> dict[str, Any]:
    return analyze_label_deterministically(
        ocr_result,
        brand_required=brand_required,
    )