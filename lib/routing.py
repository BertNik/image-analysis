# routing.py

import re
from typing import Any


def assess_ocr_usability(
    analysis,
    min_confidence=0.50,
    min_meaningful_chars=3,
):
    observations = analysis.get(
        "observations",
        [],
    )

    usable_observations = []

    for observation in observations:
        text = (
            observation.get("text")
            or ""
        ).strip()

        confidence = observation.get(
            "confidence"
        )

        meaningful_text = re.sub(
            r"[^A-Z0-9]",
            "",
            text.upper(),
        )

        # ".", "..", "E", etc. are not enough
        if len(meaningful_text) < min_meaningful_chars:
            continue

        # Ignore very low-confidence observations
        if (
            confidence is not None
            and confidence < min_confidence
        ):
            continue

        usable_observations.append(
            observation
        )

    return {
        "usable":
            len(usable_observations) > 0,

        "usable_count":
            len(usable_observations),

        "usable_observations":
            usable_observations,
    }


def route_analysis(
    analysis: dict[str, Any],
) -> dict[str, Any]:
    usability = assess_ocr_usability(
        analysis
    )

    if not usability["usable"]:
        return {
            "status": "ABSTAIN",
            "reasons": [
                "no_usable_ocr_evidence"
            ],
            "usability": usability,
        }

    missing_fields = analysis.get(
        "unresolved_required_fields",
        [],
    )

    if missing_fields:
        return {
            "status": "VERIFY",

            "reasons": [
                f"{field}_not_resolved"
                for field in missing_fields
            ],

            "usability": usability,
        }

    unresolved_roles = analysis.get(
        "unresolved_required_roles",
        [],
    )

    if unresolved_roles:
        return {
            "status":
                "SEMANTIC_FALLBACK",

            "reasons": [
                f"{role}_role_ambiguous"
                for role
                in unresolved_roles
            ],

            "usability":
                usability,
        }

    return {
        "status": "RESOLVED",
        "reasons": [],
        "usability": usability,
    }