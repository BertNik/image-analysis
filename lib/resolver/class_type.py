from __future__ import annotations

from copy import deepcopy
from math import isfinite
from typing import Any

from lib.resolver.semantic.grounding import (
    collect_grounded_entities,
    format_grounded_entity
)
from lib.resolver.semantic.extraction import ensure_extraction

MAX_CLASS_TYPE_LINES = 3

MINIMUM_HORIZONTAL_OVERLAP = 0.55

MAXIMUM_VERTICAL_GAP_RATIO = 0.85


MARKETING_ONLY_PHRASES = {
    "SMALL BATCH",
    "SMALLBATCH",
    "RESERVE",
    "SPECIAL RESERVE",
    "PREMIUM",
    "HAND CRAFTED",
    "HANDCRAFTED",
    "LIMITED EDITION",
}

CLASS_TYPE_ANCHORS = {
    "WHISKEY",
    "WHISKY",
    "BOURBON",
    "VODKA",
    "GIN",
    "RUM",
    "TEQUILA",
    "BRANDY",
    "CORDIAL",
    "LIQUEUR",
    "WINE",
    "BEER",
    "ALE",
    "LAGER",
    "STOUT",
}

def normalize_text(
    value: Any,
) -> str:
    return " ".join(
        str(value or "")
        .upper()
        .split()
    )

def normalized_words(
    value: str,
) -> set[str]:
    return {
        word.strip(
            ".,:;()[]"
        ).upper()
        for word in str(
            value or ""
        ).split()
        if word.strip()
    }

def has_class_type_anchor(
    value: str,
) -> bool:
    return bool(
        normalized_words(value)
        & CLASS_TYPE_ANCHORS
    )

def adjacent_class_type_candidates(
    observations: list[
        dict[str, Any]
    ],
    *,
    maximum_lines: int = 3,
    maximum_vertical_gap: float = 30.0,
) -> list[dict[str, Any]]:
    candidates = []

    ordered = sorted(
        observations,
        key=lambda observation: (
            observation.get(
                "box",
                [0, 0, 0, 0],
            )[1],
            observation.get(
                "box",
                [0, 0, 0, 0],
            )[0],
        ),
    )

    for start in range(
        len(ordered)
    ):
        span = []

        for end in range(
            start,
            min(
                len(ordered),
                start + maximum_lines,
            ),
        ):
            current = ordered[end]

            if span:
                previous_box = span[
                    -1
                ].get("box")

                current_box = current.get(
                    "box"
                )

                if (
                    not previous_box
                    or not current_box
                ):
                    break

                vertical_gap = (
                    float(current_box[1])
                    - float(previous_box[3])
                )

                if (
                    vertical_gap
                    > maximum_vertical_gap
                ):
                    break

            span.append(current)

            if len(span) < 2:
                continue

            value = " ".join(
                str(
                    item.get(
                        "text",
                        "",
                    )
                ).strip()
                for item in span
            )

            if not has_class_type_anchor(
                value
            ):
                continue

            candidates.append(
                {
                    "value":
                        value,

                    "observed_text":
                        "\n".join(
                            str(
                                item.get(
                                    "text",
                                    "",
                                )
                            ).strip()
                            for item in span
                        ),

                    "line_indexes": [
                        item.get("index")
                        for item in span
                    ],

                    "evidence": [
                        {
                            "line_index":
                                item.get(
                                    "index"
                                ),

                            "text":
                                item.get(
                                    "text"
                                ),

                            "confidence":
                                item.get(
                                    "confidence"
                                ),
                        }
                        for item in span
                    ],

                    "method":
                        "adjacent_class_type_candidate",
                }
            )

    return candidates





def resolve_class_type_semantically(
    model,
    schema,
    analysis,
    *,
    extraction=None,
) -> dict[str, Any] | None:
    analysis[
        "semantic_class_type_candidates"
    ] = []
        
    unresolved_roles = analysis.get(
        "unresolved_required_roles",
        [],
    )

    if (
        "class_type"
        not in unresolved_roles
    ):
        return None

    extraction = ensure_extraction(
        model,
        schema,
        analysis,
        extraction,
    )

    if extraction is None:
        return None

    observations = analysis.get(
        "observations",
        [],
    )

    entries = [
        entry
        for entry in collect_grounded_entities(
            extraction,
            "class_type",
            observations,
        )
        if not is_marketing_only_text(
            entry["grounded"].get(
                "value",
                "",
            )
        )
    ]

    if len(entries) == 1:
        result = format_grounded_entity(
            entries[0],
            method=(
                "grounded_semantic_"
                "class_type_selection"
            ),
        )

        return complete_class_type_visual_block(
            result,
            observations,
        )

    if len(entries) > 1:
        combined = (
            combine_grounded_class_type_entries(
                entries,
                observations,
            )
        )

        if combined is not None:
            return combined

        analysis[
            "semantic_class_type_candidates"
        ] = [
            format_grounded_entity(
                entry,
                method=(
                    "grounded_semantic_"
                    "class_type_candidate"
                ),
            )
            for entry in entries
        ]

        return None

    adjacent_candidates = (
        adjacent_class_type_candidates(
            observations
        )
    )

    if len(adjacent_candidates) == 1:
        candidate = dict(
            adjacent_candidates[0]
        )

        candidate["method"] = (
            "grounded_adjacent_"
            "class_type_selection"
        )

        return candidate

    if adjacent_candidates:
        analysis[
            "semantic_class_type_candidates"
        ] = adjacent_candidates

    return None

def is_marketing_only_text(
    value: Any,
) -> bool:
    return (
        normalize_text(value)
        in MARKETING_ONLY_PHRASES
    )


def get_valid_box(
    observation: dict[str, Any],
) -> tuple[
    float,
    float,
    float,
    float,
] | None:
    box = observation.get(
        "box"
    )

    if (
        not isinstance(
            box,
            (list, tuple),
        )
        or len(box) != 4
    ):
        return None

    try:
        values = tuple(
            float(value)
            for value in box
        )
    except (
        TypeError,
        ValueError,
    ):
        return None

    if not all(
        isfinite(value)
        for value in values
    ):
        return None

    x1, y1, x2, y2 = values

    if (
        x2 <= x1
        or y2 <= y1
    ):
        return None

    return values

def box_width(
    box: tuple[
        float,
        float,
        float,
        float,
    ],
) -> float:
    return box[2] - box[0]


def box_height(
    box: tuple[
        float,
        float,
        float,
        float,
    ],
) -> float:
    return box[3] - box[1]


def horizontal_overlap_ratio(
    first_box: tuple[
        float,
        float,
        float,
        float,
    ],
    second_box: tuple[
        float,
        float,
        float,
        float,
    ],
) -> float:
    overlap = max(
        0.0,
        min(
            first_box[2],
            second_box[2],
        )
        - max(
            first_box[0],
            second_box[0],
        ),
    )

    denominator = min(
        box_width(first_box),
        box_width(second_box),
    )

    if denominator <= 0:
        return 0.0

    return overlap / denominator


def vertical_gap(
    first_box: tuple[
        float,
        float,
        float,
        float,
    ],
    second_box: tuple[
        float,
        float,
        float,
        float,
    ],
) -> float:
    upper_box, lower_box = sorted(
        (
            first_box,
            second_box,
        ),
        key=lambda box: box[1],
    )

    return max(
        0.0,
        lower_box[1]
        - upper_box[3],
    )

def observations_share_visual_block(
    first: dict[str, Any],
    second: dict[str, Any],
    *,
    minimum_horizontal_overlap: float = (
        MINIMUM_HORIZONTAL_OVERLAP
    ),
    maximum_vertical_gap_ratio: float = (
        MAXIMUM_VERTICAL_GAP_RATIO
    ),
) -> bool:
    first_box = get_valid_box(
        first
    )

    second_box = get_valid_box(
        second
    )

    if (
        first_box is None
        or second_box is None
    ):
        return False

    overlap = (
        horizontal_overlap_ratio(
            first_box,
            second_box,
        )
    )

    if (
        overlap
        < minimum_horizontal_overlap
    ):
        return False

    typical_height = max(
        1.0,
        min(
            box_height(first_box),
            box_height(second_box),
        ),
    )

    gap = vertical_gap(
        first_box,
        second_box,
    )

    return (
        gap
        <= typical_height
        * maximum_vertical_gap_ratio
    )

def observations_by_index(
    observations: list[
        dict[str, Any]
    ],
) -> dict[
    int,
    dict[str, Any],
]:
    indexed = {}

    for observation in observations:
        index = observation.get(
            "index"
        )

        if not isinstance(
            index,
            int,
        ):
            continue

        # First occurrence wins if malformed OCR data
        # contains duplicate observation indexes.
        indexed.setdefault(
            index,
            observation,
        )

    return indexed


def evidence_from_observation(
    observation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "line_index":
            observation["index"],

        "text":
            str(
                observation.get(
                    "text",
                    "",
                )
                or ""
            ).strip(),

        "confidence":
            observation.get(
                "confidence",
                0.0,
            ),
    }

from copy import deepcopy
from typing import Any


MAX_CLASS_TYPE_LINES = 3


def merge_class_type_observations(
    result: dict[str, Any],
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    ordered = sorted(
        observations,
        key=lambda item: (
            get_valid_box(item)[1]
            if get_valid_box(item) is not None
            else float("inf"),
            get_valid_box(item)[0]
            if get_valid_box(item) is not None
            else float("inf"),
        ),
    )

    texts = [
        str(
            item.get(
                "text",
                "",
            )
            or ""
        ).strip()
        for item in ordered
    ]

    texts = [
        text
        for text in texts
        if text
    ]

    completed = deepcopy(
        result
    )

    completed["value"] = " ".join(
        texts
    )

    completed["observed_text"] = "\n".join(
        texts
    )

    completed["line_indexes"] = [
        item["index"]
        for item in ordered
    ]

    completed["evidence"] = [
        {
            "line_index":
                item["index"],

            "text":
                str(
                    item.get(
                        "text",
                        "",
                    )
                    or ""
                ).strip(),

            "confidence":
                item.get(
                    "confidence",
                    0.0,
                ),
        }
        for item in ordered
    ]

    completed["completion_method"] = (
        "visual_class_type_block"
    )

    return completed

def combine_grounded_class_type_entries(
    entries: list[dict[str, Any]],
    observations: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not entries:
        return None

    selected_indexes = sorted({
        index
        for entry in entries
        for index in (
            entry["grounded"].get(
                "line_indexes",
                [],
            )
        )
        if isinstance(
            index,
            int,
        )
    })

    if (
        not selected_indexes
        or len(selected_indexes)
        > MAX_CLASS_TYPE_LINES
    ):
        return None

    observations_by_index = {
        observation["index"]:
            observation
        for observation in observations
        if isinstance(
            observation.get("index"),
            int,
        )
    }

    selected = [
        observations_by_index[index]
        for index in selected_indexes
        if index in observations_by_index
    ]

    if (
        len(selected)
        != len(selected_indexes)
    ):
        return None

    selected.sort(
        key=lambda item: (
            get_valid_box(item)[1]
            if get_valid_box(item) is not None
            else float("inf")
        )
    )

    for first, second in zip(
        selected,
        selected[1:],
    ):
        if not observations_share_visual_block(
            first,
            second,
        ):
            return None

    if any(
        is_marketing_only_text(
            observation.get(
                "text",
                "",
            )
        )
        for observation in selected
    ):
        return None

    semantic_observations = []

    for entry in entries:
        for value in entry.get(
            "semantic_observations",
            [],
        ):
            if (
                value
                not in semantic_observations
            ):
                semantic_observations.append(
                    value
                )

    result = {
        "semantic_observations":
            semantic_observations,

        "grounding_method":
            "grounded_semantic_span",

        "method": (
            "grounded_semantic_"
            "class_type_selection"
        ),
    }

    return merge_class_type_observations(
        result,
        selected,
    )

def complete_class_type_visual_block(
    result: dict[str, Any] | None,
    observations: list[
        dict[str, Any]
    ],
    *,
    max_lines: int = (
        MAX_CLASS_TYPE_LINES
    ),
) -> dict[str, Any] | None:
    if result is None:
        return None

    completed = deepcopy(
        result
    )

    selected_indexes = sorted({
        index
        for index in completed.get(
            "line_indexes",
            [],
        )
        if isinstance(
            index,
            int,
        )
    })

    if not selected_indexes:
        return completed

    indexed = observations_by_index(
        observations
    )

    selected = [
        indexed[index]
        for index in selected_indexes
        if index in indexed
    ]

    if not selected:
        return completed

    while (
        len(selected)
        < max_lines
    ):
        selected.sort(
            key=lambda item:
                item["index"]
        )

        first = selected[0]
        last = selected[-1]

        previous = indexed.get(
            first["index"] - 1
        )

        following = indexed.get(
            last["index"] + 1
        )

        candidates = []

        if (
            previous is not None
            and observations_share_visual_block(
                previous,
                first,
            )
        ):
            candidates.append(
                previous
            )

        if (
            following is not None
            and observations_share_visual_block(
                last,
                following,
            )
        ):
            candidates.append(
                following
            )

        if not candidates:
            break

        candidate = min(
            candidates,
            key=lambda item: min(
                abs(
                    item["index"]
                    - first["index"]
                ),
                abs(
                    item["index"]
                    - last["index"]
                ),
            ),
        )

        text = str(
            candidate.get(
                "text",
                "",
            )
            or ""
        ).strip()

        if not text:
            break

        # Marketing text ends the visual designation.
        # Do not skip through it into later OCR fields.
        if is_marketing_only_text(
            text
        ):
            break

        if any(
            item["index"]
            == candidate["index"]
            for item in selected
        ):
            break

        selected.append(
            candidate
        )

    selected.sort(
        key=lambda item:
            item["index"]
    )

    resulting_indexes = [
        item["index"]
        for item in selected
    ]

    if (
        resulting_indexes
        == selected_indexes
    ):
        return completed

    texts = [
        str(
            item.get(
                "text",
                "",
            )
            or ""
        ).strip()
        for item in selected
    ]

    texts = [
        text
        for text in texts
        if text
    ]

    completed["value"] = " ".join(
        texts
    )

    completed[
        "observed_text"
    ] = "\n".join(
        texts
    )

    completed[
        "line_indexes"
    ] = resulting_indexes

    completed["evidence"] = [
        evidence_from_observation(
            item
        )
        for item in selected
    ]

    completed[
        "completion_method"
    ] = (
        "adjacent_visual_"
        "class_type_block"
    )

    return completed
