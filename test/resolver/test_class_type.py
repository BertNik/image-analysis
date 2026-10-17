from __future__ import annotations
from math import isfinite
from typing import Any

from test.resolver.helpers import helpers 
from lib.resolver.semantic.grounding import (
    identity_key
)

import lib.resolver.class_type as class_type

MINIMUM_HORIZONTAL_OVERLAP = 0.55

MAXIMUM_VERTICAL_GAP_RATIO = 0.85


Box = tuple[
    float,
    float,
    float,
    float,
]

def get_valid_box(
    observation: dict[str, Any],
) -> Box | None:
    """
    Return an OCR observation's validated bounding box.

    Expected format:
        [x1, y1, x2, y2]

    Invalid, nonnumeric, nonfinite, zero-size, and reversed
    boxes are rejected.
    """

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
        x1, y1, x2, y2 = (
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
        for value in (
            x1,
            y1,
            x2,
            y2,
        )
    ):
        return None

    if (
        x2 <= x1
        or y2 <= y1
    ):
        return None

    return (
        x1,
        y1,
        x2,
        y2,
    )


def box_width(
    box: Box,
) -> float:
    return (
        box[2]
        - box[0]
    )


def box_height(
    box: Box,
) -> float:
    return (
        box[3]
        - box[1]
    )


def horizontal_overlap_ratio(
    first_box: Box,
    second_box: Box,
) -> float:
    """
    Measure horizontal overlap relative to the narrower box.

    A result of:
        1.0 means the narrower box is fully covered.
        0.0 means there is no horizontal overlap.
    """

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

    narrower_width = min(
        box_width(first_box),
        box_width(second_box),
    )

    if narrower_width <= 0:
        return 0.0

    return (
        overlap
        / narrower_width
    )


def vertical_gap(
    first_box: Box,
    second_box: Box,
) -> float:
    """
    Return the vertical distance between two boxes.

    Overlapping boxes have a gap of zero.
    Input order does not matter.
    """

    upper_box, lower_box = sorted(
        (
            first_box,
            second_box,
        ),
        key=lambda box:
            box[1],
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
    """
    Determine whether two OCR observations probably belong
    to the same visual text block.

    The observations must:

    1. Have valid bounding boxes.
    2. Have sufficient horizontal overlap.
    3. Have a vertical gap small enough relative to the
       height of the smaller text box.

    This function uses only layout geometry. Semantic
    validation and marketing-text rejection belong in the
    class/type completion function.
    """

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

    overlap_ratio = (
        horizontal_overlap_ratio(
            first_box,
            second_box,
        )
    )

    if (
        overlap_ratio
        < minimum_horizontal_overlap
    ):
        return False

    reference_height = max(
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

    allowed_gap = (
        reference_height
        * maximum_vertical_gap_ratio
    )

    return (
        gap
        <= allowed_gap
    )

def test_adjacent_class_type_lines_share_visual_block():
    first = {
        "index": 3,
        "text":
            "KENTUCKY STRAIGHT",
        "confidence":
            0.9995002746582031,
        "box": [
            287,
            288,
            613,
            311,
        ],
    }

    second = {
        "index": 4,
        "text":
            "BOURBON WHISKEY",
        "confidence":
            0.999434232711792,
        "box": [
            248,
            330,
            651,
            359,
        ],
    }

    assert (
        observations_share_visual_block(
            first,
            second,
        )
        is True
    )

def test_completes_kentucky_straight_bourbon_whiskey_block():
    observations = [
        {
            "index": 3,
            "text":
                "KENTUCKY STRAIGHT",
            "confidence": 0.99,
            "box": [
                164,
                874,
                860,
                932,
            ],
        },
        {
            "index": 4,
            "text":
                "BOURBON WHISKEY",
            "confidence": 0.99,
            "box": [
                79,
                902,
                954,
                1066,
            ],
        },
        {
            "index": 5,
            "text":
                "SMALLBATCH",
            "confidence": 0.99,
            "box": [
                317,
                1069,
                708,
                1108,
            ],
        },
    ]

    original = {
        "value":
            "KENTUCKY STRAIGHT",

        "observed_text":
            "KENTUCKY STRAIGHT",

        "line_indexes": [
            3,
        ],

        "evidence": [
            {
                "line_index": 3,
                "text":
                    "KENTUCKY STRAIGHT",
                "confidence": 0.99,
            }
        ],

        "semantic_observations": [
            "KENTUCKY STRAIGHT",
        ],

        "grounding_method":
            "grounded_semantic_span",

        "method": (
            "grounded_semantic_"
            "class_type_selection"
        ),
    }

    result = (
        class_type
        .complete_class_type_visual_block(
            original,
            observations,
        )
    )

    assert result is not None

    assert result["value"] == (
        "KENTUCKY STRAIGHT "
        "BOURBON WHISKEY"
    )

    assert result[
        "observed_text"
    ] == (
        "KENTUCKY STRAIGHT\n"
        "BOURBON WHISKEY"
    )

    assert result[
        "line_indexes"
    ] == [
        3,
        4,
    ]

    assert [
        item["text"]
        for item in result["evidence"]
    ] == [
        "KENTUCKY STRAIGHT",
        "BOURBON WHISKEY",
    ]

def test_does_not_absorb_marketing_text():
    observations = [
        {
            "index": 3,
            "text":
                "BOURBON WHISKEY",
            "confidence": 0.99,
            "box": [
                100,
                100,
                500,
                150,
            ],
        },
        {
            "index": 4,
            "text":
                "SMALL BATCH",
            "confidence": 0.99,
            "box": [
                150,
                152,
                450,
                190,
            ],
        },
    ]

    original = {
        "value":
            "BOURBON WHISKEY",

        "observed_text":
            "BOURBON WHISKEY",

        "line_indexes": [
            3,
        ],

        "evidence": [
            {
                "line_index": 3,
                "text":
                    "BOURBON WHISKEY",
                "confidence": 0.99,
            }
        ],
    }

    result = (
        class_type
        .complete_class_type_visual_block(
            original,
            observations,
        )
    )

    assert result is not None

    assert result["value"] == (
        "BOURBON WHISKEY"
    )

    assert result[
        "line_indexes"
    ] == [
        3,
    ]

def test_class_type_extends_grounded_anchor_to_adjacent_designation():
    analysis = {
        "ocr_text": (
            "KENTUCKY STRAIGHT\n"
            "BOURBON WHISKEY"
        ),
        "observations": [
            {
                "index": 3,
                "text":
                    "KENTUCKY STRAIGHT",
                "confidence":
                    0.9995002746582031,
                "box": [
                    287,
                    288,
                    613,
                    311,
                ],
            },
            {
                "index": 4,
                "text":
                    "BOURBON WHISKEY",
                "confidence":
                    0.999434232711792,
                "box": [
                    248,
                    330,
                    651,
                    359,
                ],
            },
        ],
        "unresolved_required_roles": [
            "class_type",
        ],
    }

    extraction = {
        "entities": {
            "class_type": [
                "KENTUCKY STRAIGHT",
            ],
        },
    }

    result = (
        class_type.resolve_class_type_semantically(
            model=None,
            schema=None,
            analysis=analysis,
            extraction=extraction,
        )
    )

    assert result is not None

    assert result["value"] == (
        "KENTUCKY STRAIGHT "
        "BOURBON WHISKEY"
    )

    assert result[
        "observed_text"
    ] == (
        "KENTUCKY STRAIGHT\n"
        "BOURBON WHISKEY"
    )

    assert result[
        "line_indexes"
    ] == [
        3,
        4,
    ]

    assert [
        evidence["text"]
        for evidence
        in result["evidence"]
    ] == [
        "KENTUCKY STRAIGHT",
        "BOURBON WHISKEY",
    ]

def test_combines_visual_lines_across_interleaved_ocr_indexes():
    observations = [
        {
            "index": 20,
            "text":
                "KENTUCKY STRAIGHT",
            "confidence": 0.99,
            "box": [
                119,
                710,
                413,
                736,
            ],
        },
        {
            "index": 21,
            "text":
                "GOV_WARNING 0.97",
            "confidence": 0.98,
            "box": [
                604,
                707,
                748,
                725,
            ],
        },
        {
            "index": 22,
            "text":
                "GOVERNMENT WARNING:",
            "confidence": 0.99,
            "box": [
                617,
                730,
                808,
                748,
            ],
        },
        {
            "index": 23,
            "text":
                "BOURBON WHISKEY",
            "confidence": 0.99,
            "box": [
                127,
                747,
                408,
                777,
            ],
        },
    ]

    entries = [
        {
            "grounded": {
                "value":
                    "KENTUCKY STRAIGHT",
                "line_indexes": [20],
            },
            "semantic_observations": [
                "KENTUCKY STRAIGHT",
            ],
        },
        {
            "grounded": {
                "value":
                    "BOURBON WHISKEY",
                "line_indexes": [23],
            },
            "semantic_observations": [
                "BOURBON WHISKEY",
            ],
        },
    ]

    result = (
        class_type.combine_grounded_class_type_entries(
            entries,
            observations,
        )
    )

    assert result is not None

    assert result["line_indexes"] == [
        20,
        23,
    ]

    assert result["value"] == (
        "KENTUCKY STRAIGHT "
        "BOURBON WHISKEY"
    )
    
def test_combines_multiple_semantic_class_type_parts():
    analysis = {
        "ocr_text": (
            "KENTUCKY STRAIGHT\n"
            "BOURBON WHISKEY"
        ),
        "observations": [
            {
                "index": 4,
                "text":
                    "KENTUCKY STRAIGHT",
                "confidence": 0.99,
                "box": [
                    183,
                    921,
                    864,
                    979,
                ],
            },
            {
                "index": 5,
                "text":
                    "BOURBON WHISKEY",
                "confidence": 0.99,
                "box": [
                    95,
                    984,
                    941,
                    1099,
                ],
            },
        ],
        "unresolved_required_roles": [
            "class_type",
        ],
    }

    extraction = {
        "entities": {
            "class_type": [
                "KENTUCKY STRAIGHT",
                "BOURBON WHISKEY",
            ],
        },
    }

    result = (
        class_type.resolve_class_type_semantically(
            model=None,
            schema=None,
            analysis=analysis,
            extraction=extraction,
        )
    )

    assert result is not None

    assert result["value"] == (
        "KENTUCKY STRAIGHT "
        "BOURBON WHISKEY"
    )

    assert result["line_indexes"] == [
        4,
        5,
    ]

    
def test_class_type_multiple_semantic_entries_remain_unresolved(
    monkeypatch,
):
    analysis = {
        "unresolved_required_roles": [
            "class_type",
        ],
        "observations": [
            {
                "index": 0,
                "text": "BOURBON WHISKEY",
            },
        ],
    }

    entries = [
        {
            "grounded": {
                "value":
                    "BOURBON WHISKEY",
                "observed_text":
                    "BOURBON WHISKEY",
                "line_indexes": [0],
                "evidence": [],
                "method":
                    "grounded_semantic_span",
            },
            "semantic_observations": [
                "BOURBON WHISKEY",
            ],
        },
        {
            "grounded": {
                "value":
                    "STRAIGHT WHISKEY",
                "observed_text":
                    "STRAIGHT WHISKEY",
                "line_indexes": [1],
                "evidence": [],
                "method":
                    "grounded_semantic_span",
            },
            "semantic_observations": [
                "STRAIGHT WHISKEY",
            ],
        },
    ]

    monkeypatch.setattr(
        class_type,
        "collect_grounded_entities",
        lambda *args, **kwargs:
            entries,
    )

    result = (
        class_type.resolve_class_type_semantically(
            model=object(),
            schema=object(),
            analysis=analysis,
            extraction={
                "entities": {},
            },
        )
    )

    assert result is None

    candidates = analysis[
        "semantic_class_type_candidates"
    ]

    assert len(candidates) == 2

    assert candidates[0]["value"] == (
        "BOURBON WHISKEY"
    )

    assert candidates[1]["value"] == (
        "STRAIGHT WHISKEY"
    )

    assert candidates[0]["method"] == (
        "grounded_semantic_"
        "class_type_candidate"
    )

def test_class_type_uses_single_adjacent_candidate(
    monkeypatch,
):
    observations = [
        {
            "index": 3,
            "text":
                "KENTUCKY STRAIGHT",
        },
        {
            "index": 4,
            "text":
                "BOURBON WHISKEY",
        },
    ]

    analysis = {
        "unresolved_required_roles": [
            "class_type",
        ],
        "observations":
            observations,
    }

    # No semantic class/type candidates.
    monkeypatch.setattr(
        
            class_type,
            "collect_grounded_entities"
        ,
        lambda *args, **kwargs:
            [],
    )

    adjacent_candidate = {
        "value": (
            "KENTUCKY STRAIGHT "
            "BOURBON WHISKEY"
        ),

        "observed_text": (
            "KENTUCKY STRAIGHT\n"
            "BOURBON WHISKEY"
        ),

        "line_indexes": [
            3,
            4,
        ],

        "evidence": [],
    }

    monkeypatch.setattr(

            class_type,
            "adjacent_class_type_candidates"
        ,
        lambda observations:
            [
                adjacent_candidate,
            ],
    )

    result = (
        class_type.resolve_class_type_semantically(
            model=object(),
            schema=object(),
            analysis=analysis,
            extraction={
                "entities": {},
            },
        )
    )

    assert result is not None

    assert result["value"] == (
        "KENTUCKY STRAIGHT "
        "BOURBON WHISKEY"
    )

    assert result[
        "line_indexes"
    ] == [
        3,
        4,
    ]

    assert result["method"] == (
        "grounded_adjacent_"
        "class_type_selection"
    )

def test_class_type_multiple_adjacent_candidates_remain_unresolved(
    monkeypatch,
):
    analysis = {
        "unresolved_required_roles": [
            "class_type",
        ],
        "observations": [],
    }

    monkeypatch.setattr(
            class_type,
            "collect_grounded_entities"
        ,
        lambda *args, **kwargs:
            [],
    )

    candidates = [
        {
            "value":
                "KENTUCKY STRAIGHT "
                "BOURBON WHISKEY",

            "line_indexes":
                [3, 4],

            "method":
                "adjacent_class_type_candidate",
        },
        {
            "value":
                "STRAIGHT BOURBON "
                "WHISKEY",

            "line_indexes":
                [4, 5],

            "method":
                "adjacent_class_type_candidate",
        },
    ]

    monkeypatch.setattr(
        
            class_type,
            "adjacent_class_type_candidates"
        ,
        lambda observations:
            candidates,
    )

    result = (
        class_type.resolve_class_type_semantically(
            model=object(),
            schema=object(),
            analysis=analysis,
            extraction={
                "entities": {},
            },
        )
    )

    assert result is None

    assert analysis[
        "semantic_class_type_candidates"
    ] == candidates

def test_class_type_no_adjacent_candidates_remains_unresolved(
    monkeypatch,
):
    analysis = {
        "unresolved_required_roles": [
            "class_type",
        ],
        "observations": [],
    }

    monkeypatch.setattr(
        class_type,
        "collect_grounded_entities",
        lambda *args, **kwargs: [],
    )

    monkeypatch.setattr(
        class_type,
        "adjacent_class_type_candidates",
        lambda observations: [],
    )

    result = (
        class_type
        .resolve_class_type_semantically(
            model=object(),
            schema=object(),
            analysis=analysis,
            extraction={
                "entities": {},
            },
        )
    )

    assert result is None

    assert analysis[
        "semantic_class_type_candidates"
    ] == []

def test_adjacent_class_type_lines_are_joined():
    observations = [
        {
            "index": 3,
            "text":
                "KENTUCKY STRAIGHT",
            "confidence": 0.99,
            "box": [
                287,
                288,
                613,
                311,
            ],
        },
        {
            "index": 4,
            "text":
                "BOURBON WHISKEY",
            "confidence": 0.99,
            "box": [
                248,
                330,
                651,
                359,
            ],
        },
    ]

    candidates = (
        class_type.adjacent_class_type_candidates(
            observations
        )
    )

    assert len(candidates) == 1

    assert candidates[0]["value"] == (
        "KENTUCKY STRAIGHT "
        "BOURBON WHISKEY"
    )

    assert candidates[0][
        "line_indexes"
    ] == [
        3,
        4,
    ]

def test_marketing_term_is_not_class_type():
    observations = [
        {
            "index": 0,
            "text": "SMALL BATCH",
            "confidence": 0.99,
            "box": [
                100,
                100,
                300,
                125,
            ],
        },
        {
            "index": 1,
            "text": "HANDCRAFTED",
            "confidence": 0.99,
            "box": [
                100,
                130,
                300,
                155,
            ],
        },
    ]

    assert (
        class_type.adjacent_class_type_candidates(
            observations
        )
        == []
    )

def test_grounded_multiline_class_type_resolves():
    model = helpers.FakeModel(
        {
            "entities": {
                "class_type": [
                    (
                        "KENTUCKY STRAIGHT "
                        "BOURBON WHISKEY"
                    ),
                ],
            }
        }
    )

    analysis = {
        "ocr_text": (
            "KENTUCKY STRAIGHT\n"
            "BOURBON WHISKEY"
        ),

        "observations": [
            {
                "index": 3,
                "text":
                    "KENTUCKY STRAIGHT",
                "confidence": 0.99,
                "box": None,
            },
            {
                "index": 4,
                "text":
                    "BOURBON WHISKEY",
                "confidence": 0.98,
                "box": None,
            },
        ],

        "resolved": {
            "class_type": None,
        },

        "unresolved_required_fields":
            [],

        "unresolved_required_roles": [
            "class_type",
        ],

        "unresolved": {
            "class_type": {
                "reason":
                    "class_type_requires_semantic_resolution",
            }
        },

        "semantic_fields": [],
    }

    result = (
        class_type.resolve_class_type_semantically(
            model,
            schema="schema",
            analysis=analysis,
        )
    )

    assert result is not None

    assert result["line_indexes"] == [
        3,
        4,
    ]

    assert (
        identity_key(
            result["value"]
        )
        ==
        "KENTUCKYSTRAIGHT"
        "BOURBONWHISKEY"
    )

    assert (
        result["method"]
        ==
        "grounded_semantic_class_type_selection"
    )
