from lib.resolver.semantic.country import (
    resolve_country_of_origin_semantically,
)

from test.resolver.helpers.helpers import (
    FakeModel,
)

def test_country_of_origin_resolves():
    model = FakeModel(
        {
            "entities": {
                "country_of_origin": [
                    "PRODUCT OF THE UNITED STATES",
                ],
            },
        }
    )

    analysis = {
        "ocr_text": (
            "PRODUCT OF THE UNITED STATES"
        ),

        "observations": [
            {
                "index": 10,
                "text": (
                    "PRODUCT OF "
                    "THE UNITED STATES"
                ),
                "confidence": 0.99,
                "box": [
                    10,
                    10,
                    300,
                    30,
                ],
            },
        ],

        # Required by the resolver's entry guard.
        "unresolved_required_country_of_origin": [
            "country_of_origin",
        ],
    }

    result = (
        resolve_country_of_origin_semantically(
            model,
            schema="schema",
            analysis=analysis,
        )
    )

    assert result is not None

    assert result["value"] == (
        "PRODUCT OF THE UNITED STATES"
    )

    assert result["observed_text"] == (
        "PRODUCT OF THE UNITED STATES"
    )

    assert result["line_indexes"] == [
        10,
    ]

    assert result["method"] == (
        "grounded_semantic_"
        "country_of_origin_selection"
    )

def test_country_of_origin_returns_none_when_not_extracted():
    model = FakeModel(
        {
            "entities": {},
        }
    )

    analysis = {
        "ocr_text":
            "KENTUCKY STRAIGHT BOURBON WHISKEY",

        "observations": [
            {
                "index": 0,
                "text": (
                    "KENTUCKY STRAIGHT "
                    "BOURBON WHISKEY"
                ),
                "confidence": 0.99,
                "box": None,
            },
        ],
    }

    result = (
        resolve_country_of_origin_semantically(
            model,
            schema="schema",
            analysis=analysis,
        )
    )

    assert result is None

def test_multiple_country_statements_remain_unresolved():
    model = FakeModel(
        {
            "entities": {
                "country_of_origin": [
                    "PRODUCT OF FRANCE",
                    "PRODUCT OF ITALY",
                ],
            },
        }
    )

    analysis = {
        "ocr_text": (
            "PRODUCT OF FRANCE\n"
            "PRODUCT OF ITALY"
        ),

        "observations": [
            {
                "index": 0,
                "text": "PRODUCT OF FRANCE",
                "confidence": 0.99,
                "box": None,
            },
            {
                "index": 1,
                "text": "PRODUCT OF ITALY",
                "confidence": 0.99,
                "box": None,
            },
        ],

        # Required by the resolver guard.
        "unresolved_required_country_of_origin": [
            "country_of_origin",
        ],
    }

    result = (
        resolve_country_of_origin_semantically(
            model,
            schema="schema",
            analysis=analysis,
        )
    )

    assert result is None

    candidates = analysis[
        "semantic_country_of_origin_candidates"
    ]

    assert len(candidates) == 2

    assert {
        candidate["value"]
        for candidate in candidates
    } == {
        "PRODUCT OF FRANCE",
        "PRODUCT OF ITALY",
    }

def test_semantic_country_of_origin_returns_candidate():
    model = FakeModel(
        {
            "entities": {
                "country_of_origin": [
                    "FRANCE",
                ],
            },
        }
    )

    analysis = {
        "ocr_text": (
            "MAISON BELLE\n"
            "COGNAC\n"
            "PRODUCT OF FRANCE\n"
            "IMPORTED BY\n"
            "ATLANTIC IMPORTS LLC"
        ),
        "observations": [
            {
                "index": 0,
                "text": "MAISON BELLE",
                "confidence": 0.99,
            },
            {
                "index": 1,
                "text": "COGNAC",
                "confidence": 0.99,
            },
            {
                "index": 2,
                "text": "PRODUCT OF FRANCE",
                "confidence": 0.99,
            },
            {
                "index": 3,
                "text": "IMPORTED BY",
                "confidence": 0.99,
            },
            {
                "index": 4,
                "text": "ATLANTIC IMPORTS LLC",
                "confidence": 0.99,
            },
        ],
        "unresolved_required_country_of_origin": [
            "country_of_origin",
        ],
    }

    result = (
        resolve_country_of_origin_semantically(
            model,
            schema=None,
            analysis=analysis,
        )
    )

    assert result is None

    candidates = analysis[
        "semantic_country_of_origin_candidates"
    ]

    assert candidates == [
        {
            "value": "FRANCE",
            "semantic_observations": [
                "FRANCE",
            ],
            "method": (
                "semantic_country_of_origin_candidate"
            ),
        },
    ]