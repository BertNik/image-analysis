from __future__ import annotations

import re
from typing import Any, Optional

CLASS_TYPE_PATTERNS = [
    re.compile(
        r"\bKENTUCKY\s+STRAIGHT\s+BOURBON\s+WHISK(?:E)?Y\b",
        re.I,
    ),
    re.compile(
        r"\bSTRAIGHT\s+BOURBON\s+WHISK(?:E)?Y\b",
        re.I,
    ),
    re.compile(
        r"\bBOURBON\s+WHISK(?:E)?Y\b",
        re.I,
    ),
    re.compile(
        r"\bSTRAIGHT\s+RYE\s+WHISK(?:E)?Y\b",
        re.I,
    ),
    re.compile(
        r"\bRYE\s+WHISK(?:E)?Y\b",
        re.I,
    ),
    re.compile(r"\bVODKA\b", re.I),
    re.compile(r"\bGIN\b", re.I),
    re.compile(r"\bRUM\b", re.I),
    re.compile(r"\bBRANDY\b", re.I),
]

ADDRESS_PATTERN = re.compile(
    r"""
    \b
    [A-Z][A-Z .'-]+
    ,
    \s*
    [A-Z]{2}
    (?:
        \s+\d{5}(?:-\d{4})?
    )?
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)


ESTABLISHED_PATTERN = re.compile(
    r"\bEST\.?\s*\d{2,4}\b",
    re.IGNORECASE,
)


MARKETING_PATTERN = re.compile(
    r"""
    \b
    (
        SMALL\s*BATCH
        |
        SINGLE\s+BARREL
        |
        LIMITED\s+EDITION
        |
        HANDCRAFTED
        |
        RESERVE
        |
        SELECT
    )
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)

BUSINESS_MARKER_PATTERN = re.compile(
    r"""
    \b
    (?P<role>
        DISTILLED
        |
        BOTTLED
        |
        IMPORTED
        |
        PRODUCED
        |
        PACKED
        |
        FILLED
    )
    (?:
        \s+AND\s+
        (?P<secondary_role>
            DISTILLED
            |
            BOTTLED
            |
            IMPORTED
            |
            PRODUCED
            |
            PACKED
            |
            FILLED
        )
    )?
    \s+BY
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)

ABV_PATTERN = re.compile(
    r"""
    \b
    (?P<value>\d+(?:\.\d+)?)
    \s*%
    \s*
    ALC(?:OHOL)?
    \.?
    \s*
    (?:/|BY)
    \s*
    VOL(?:UME)?
    \.?
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)


PROOF_PATTERN = re.compile(
    r"""
    \(?
    \s*
    (?P<value>\d+(?:\.\d+)?)
    \s*
    PROOF
    \s*
    \)?
    """,
    re.IGNORECASE | re.VERBOSE,
)


PROOF_LIKE_PATTERN = re.compile(
    r"""
    \(?
    \s*
    (?P<value>\d+(?:\.\d+)?)
    \s*
    (?P<token>
        [A-Z0-9]*PR[A-Z0-9]*
        |
        [A-Z0-9]*ROOF[A-Z0-9]*
    )
    \s*
    \)?
    """,
    re.IGNORECASE | re.VERBOSE,
)


NET_CONTENTS_PATTERN = re.compile(
    r"""
    \b
    (?P<value>\d+(?:\.\d+)?)
    \s*
    (?P<unit>
        ML
        |
        MILLILITERS?
        |
        L
        |
        LITERS?
        |
        LITRES?
    )
    \.?
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)

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

        if (
            measurements["alcohol_content"]
            is None
        ):
            match = ABV_PATTERN.search(
                text
            )

            if match:
                measurements[
                    "alcohol_content"
                ] = {
                    "value": float(
                        match.group(
                            "value"
                        )
                    ),
                    "unit":
                        "percent_abv",
                    "observed_text":
                        text,
                    "line_indexes": [
                        item["index"]
                    ],
                    "confidence":
                        item["confidence"],
                    "method":
                        "regex",
                }

        if (
            measurements["net_contents"]
            is None
        ):
            match = (
                NET_CONTENTS_PATTERN
                .search(text)
            )

            if match:
                raw_unit = (
                    match.group(
                        "unit"
                    ).upper()
                )

                unit = (
                    "mL"
                    if raw_unit.startswith(
                        "M"
                    )
                    else "L"
                )

                measurements[
                    "net_contents"
                ] = {
                    "value": float(
                        match.group(
                            "value"
                        )
                    ),
                    "unit": unit,
                    "observed_text":
                        text,
                    "line_indexes": [
                        item["index"]
                    ],
                    "confidence":
                        item["confidence"],
                    "method":
                        "regex",
                }

        if (
            measurements["proof"]
            is None
        ):
            match = PROOF_PATTERN.search(
                text
            )

            if match:
                measurements[
                    "proof"
                ] = {
                    "value": float(
                        match.group(
                            "value"
                        )
                    ),
                    "unit": "proof",
                    "observed_text":
                        text,
                    "line_indexes": [
                        item["index"]
                    ],
                    "confidence":
                        item["confidence"],
                    "method":
                        "regex",
                }

            else:
                proof_like_match = (
                    PROOF_LIKE_PATTERN
                    .search(text)
                )

                if proof_like_match:
                    measurements[
                        "unresolved_measurements"
                    ].append(
                        {
                            "type":
                                "proof_like",

                            "candidate_value":
                                float(
                                    proof_like_match
                                    .group(
                                        "value"
                                    )
                                ),

                            "observed_text":
                                text,

                            "line_indexes": [
                                item[
                                    "index"
                                ]
                            ],

                            "confidence":
                                item[
                                    "confidence"
                                ],
                        }
                    )

    return measurements



def resolve_proof_constraint(
    measurements: dict[str, Any],
    tolerance: float = 0.01,
) -> Optional[dict[str, Any]]:
    if (
        measurements["proof"]
        is not None
    ):
        return measurements["proof"]

    alcohol = measurements[
        "alcohol_content"
    ]

    if alcohol is None:
        return None

    expected_proof = (
        alcohol["value"] * 2
    )

    for candidate in measurements[
        "unresolved_measurements"
    ]:
        if (
            candidate["type"]
            != "proof_like"
        ):
            continue

        candidate_value = (
            candidate[
                "candidate_value"
            ]
        )

        if (
            abs(
                candidate_value
                - expected_proof
            )
            <= tolerance
        ):
            return {
                "value":
                    candidate_value,

                "unit":
                    "proof",

                "observed_text":
                    candidate[
                        "observed_text"
                    ],

                "line_indexes":
                    candidate[
                        "line_indexes"
                    ],

                "confidence":
                    candidate[
                        "confidence"
                    ],

                "method":
                    "cross_field_constraint",

                "evidence": {
                    "abv":
                        alcohol["value"],

                    "expected_proof":
                        expected_proof,
                },
            }

    return None
