from lib.case.reconcile import collect_case_candidates 

def test_collect_case_candidates_filters_deduplicates_and_adds_provenance():
    image_results = [
        {
            "image_index": 0,
            "filename": "front.png",
            "analysis": {
                "semantic_class_type_candidates": [
                    # Accepted.
                    {
                        "value": (
                            "KENTUCKY STRAIGHT "
                            "BOURBON WHISKEY"
                        ),
                        "line_indexes": [
                            3,
                            4,
                        ],
                        "method":
                            "candidate_one",
                    },

                    # Exact duplicate:
                    # same value,
                    # same image,
                    # same line indexes.
                    {
                        "value": (
                            "KENTUCKY STRAIGHT "
                            "BOURBON WHISKEY"
                        ),
                        "line_indexes": [
                            3,
                            4,
                        ],
                        "method":
                            "duplicate",
                    },

                    # Exercises:
                    #
                    # if not value:
                    #     continue
                    {
                        "value": None,
                        "line_indexes": [
                            5,
                        ],
                    },

                    # Also exercises falsy value.
                    {
                        "value": "",
                        "line_indexes": [
                            6,
                        ],
                    },
                ],
            },
        },

        {
            "image_index": 1,
            "filename": "back.png",
            "analysis": {
                "semantic_class_type_candidates": [
                    # Same value and lines,
                    # but different image.
                    # Therefore unique.
                    {
                        "value": (
                            "KENTUCKY STRAIGHT "
                            "BOURBON WHISKEY"
                        ),
                        "line_indexes": [
                            3,
                            4,
                        ],
                    },

                    # Same value and image,
                    # but different lines.
                    # Therefore unique.
                    {
                        "value": (
                            "KENTUCKY STRAIGHT "
                            "BOURBON WHISKEY"
                        ),
                        "line_indexes": [
                            7,
                            8,
                        ],
                    },
                ],
            },
        },
    ]

    result = collect_case_candidates(
        image_results,
        "class_type",
    )

    assert len(result) == 3

    assert result[0]["value"] == (
        "KENTUCKY STRAIGHT "
        "BOURBON WHISKEY"
    )

    assert result[0]["image_index"] == 0
    assert result[0]["filename"] == "front.png"

    assert result[0]["line_indexes"] == [
        3,
        4,
    ]

    assert result[1]["image_index"] == 1
    assert result[1]["filename"] == "back.png"

    assert result[1]["line_indexes"] == [
        3,
        4,
    ]

    assert result[2]["image_index"] == 1
    assert result[2]["filename"] == "back.png"

    assert result[2]["line_indexes"] == [
        7,
        8,
    ]

def test_collect_case_candidates_deduplicates_by_value_image_and_lines():
    image_results = [
        {
            "image_index": 0,
            "filename": "front.png",
            "analysis": {
                "semantic_class_type_candidates": [
                    {
                        "value":
                            "BOURBON WHISKEY",
                        "line_indexes": [
                            3,
                            4,
                        ],
                    },
                    {
                        "value":
                            "BOURBON WHISKEY",
                        "line_indexes": [
                            3,
                            4,
                        ],
                    },
                    {
                        "value":
                            "BOURBON WHISKEY",
                        "line_indexes": [
                            7,
                            8,
                        ],
                    },
                ],
            },
        },
    ]

    result = collect_case_candidates(
        image_results,
        "class_type",
    )

    assert len(result) == 2

    assert [
        item["line_indexes"]
        for item in result
    ] == [
        [3, 4],
        [7, 8],
    ]

def test_collect_semantic_candidates_handles_missing_candidate_list():
    image_results = [
        {
            "image_index": 0,
            "filename": "front.png",
            "analysis": {},
        },
    ]

    result = collect_case_candidates(
        image_results,
        "semantic_class_type_candidates",
    )

    assert result == []

def test_collect_case_candidates_deduplicates_by_value_image_and_lines():
    image_results = [
        {
            "image_index": 0,
            "filename": "front.png",
            "analysis": {
                "semantic_class_type_candidates": [
                    {
                        "value":
                            "BOURBON WHISKEY",
                        "line_indexes": [
                            3,
                            4,
                        ],
                    },
                    {
                        "value":
                            "BOURBON WHISKEY",
                        "line_indexes": [
                            3,
                            4,
                        ],
                    },
                    {
                        "value":
                            "BOURBON WHISKEY",
                        "line_indexes": [
                            7,
                            8,
                        ],
                    },
                ],
            },
        },
    ]

    result = collect_case_candidates(
        image_results,
        "class_type",
    )

    assert len(result) == 2

    assert [
        item["line_indexes"]
        for item in result
    ] == [
        [3, 4],
        [7, 8],
    ]
