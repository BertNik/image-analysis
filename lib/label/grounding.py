from __future__ import annotations

import re
from typing import Any, Optional


def identity_key(
    text: Optional[str],
) -> str:
    if not text:
        return ""

    return re.sub(
        r"[^A-Z0-9]",
        "",
        str(text).upper(),
    )


def _build_grounded_result(
    observations,
):
    observed_text = " ".join(
        str(
            observation.get(
                "text",
                "",
            )
        ).strip()
        for observation
        in observations
    ).strip()

    return {
        "value":
            observed_text,

        "observed_text":
            observed_text,

        "line_indexes": [
            observation["index"]
            for observation
            in observations
        ],

        "evidence": [
            {
                "line_index":
                    observation["index"],

                "text":
                    observation.get(
                        "text",
                        "",
                    ),

                "confidence":
                    observation.get(
                        "confidence"
                    ),
            }
            for observation
            in observations
        ],

        "method":
            "grounded_semantic_span",
    }

def ground_semantic_span(
    extracted_value: str,
    observations: list[
        dict[str, Any]
    ],
) -> Optional[dict[str, Any]]:
    """
    Ground a semantic extraction in exact OCR evidence.

    Accepts:
      - one exact OCR observation
      - one exact contiguous sequence of OCR observations

    Rejects:
      - partial substring matches
      - reordered observations
      - text absent from OCR
    """

    extracted_key = identity_key(
        extracted_value
    )

    if not extracted_key:
        return None

    exact_matches = [
        observation
        for observation in observations
        if (
            identity_key(
                observation.get(
                    "text",
                    "",
                )
            )
            == extracted_key
        )
    ]

    if exact_matches:
        first = exact_matches[0]

        return {
            "value":
                first.get(
                    "text",
                    "",
                ),

            "observed_text":
                first.get(
                    "text",
                    "",
                ),

            "line_indexes": [
                observation["index"]
                for observation
                in exact_matches
            ],

            "evidence": [
                {
                    "line_index":
                        observation["index"],

                    "text":
                        observation.get(
                            "text",
                            "",
                        ),

                    "confidence":
                        observation.get(
                            "confidence"
                        ),
                }
                for observation
                in exact_matches
            ],

            "method":
                "grounded_semantic_span",
        }

    for start in range(
        len(observations)
    ):
        combined_key = ""

        for end in range(
            start,
            len(observations),
        ):
            combined_key += identity_key(
                observations[end].get(
                    "text",
                    "",
                )
            )

            if combined_key == extracted_key:
                return (
                    _build_grounded_result(
                        observations[
                            start:
                            end + 1
                        ]
                    )
                )

            if len(combined_key) >= len(
                extracted_key
            ):
                break

    return None
