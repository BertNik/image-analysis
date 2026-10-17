from lib.label.grounding import (
    identity_key,
)
from lib.resolver.semantic.brand import (
    resolve_brand_semantically,
)

from test.resolver.helpers.helpers import (
    FakeModel,
    candidate,
    semantic_analysis,
)

def test_semantic_brand_must_be_grounded_in_ocr():
    model = FakeModel(
        {
            "entities": {
                "brand_name": [
                    "JAMESON",
                ],
            }
        }
    )

    analysis = semantic_analysis(
        candidates=[
            candidate(
                "JAMESON",
                confidence=0.99,
            )
        ],
        ocr_text="JAMESON",
    )

    # Candidate exists, but its OCR observation is removed.
    analysis["observations"] = []

    result = (
        resolve_brand_semantically(
            model,
            schema="schema",
            analysis=analysis,
        )
    )

    assert result is None

def test_grounded_semantic_brand_is_accepted():
    model = FakeModel(
        {
            "entities": {
                "brand_name": [
                    "JAMESON",
                ],
            }
        }
    )

    result = (
        resolve_brand_semantically(
            model,
            schema="schema",
            analysis=semantic_analysis(
                candidates=[
                    candidate(
                        "JAMESON",
                        confidence=0.99,
                        line_indexes=[1],
                    )
                ],
                ocr_text=(
                    "JAMESON\n"
                    "IRISH WHISKEY"
                ),
            ),
        )
    )

    assert result is not None
    assert result["value"] == "JAMESON"
    assert result["line_indexes"] == [1]

    assert (
        result["grounding_method"]
        == "grounded_semantic_span"
    )

    assert (
        result["method"]
        ==
        "grounded_semantic_brand_selection"
    )

def test_expanded_semantic_span_may_ground_to_contiguous_ocr():
    model = FakeModel(
        {
            "entities": {
                "brand_name": [
                    "RIVEROAK DISTILLING CO.",
                ],
            }
        }
    )

    analysis = semantic_analysis(
        candidates=[],
        ocr_text=(
            "RIVEROAK\n"
            "DISTILLING CO."
        ),
    )

    analysis["observations"] = [
        {
            "index": 1,
            "text": "RIVEROAK",
            "confidence": 0.99,
            "box": None,
        },
        {
            "index": 2,
            "text": "DISTILLING CO.",
            "confidence": 0.98,
            "box": None,
        },
    ]

    result = (
        resolve_brand_semantically(
            model,
            schema="schema",
            analysis=analysis,
        )
    )

    assert result is not None

    assert (
        identity_key(
            result["value"]
        )
        == "RIVEROAKDISTILLINGCO"
    )

    assert result["line_indexes"] == [
        1,
        2,
    ]

    assert (
        result["grounding_method"]
        == "grounded_semantic_span"
    )


def test_resolve_brand_semantically_skips_when_brand_role_not_unresolved():
    model = FakeModel(
        {
            "entities": {
                "brand_name": [
                    "JAMESON",
                ]
            }
        }
    )

    result = (
        resolve_brand_semantically(
            model,
            schema="schema",
            analysis=semantic_analysis(
                candidates=[
                    candidate("JAMESON"),
                ],
                unresolved_roles=[],
            ),
        )
    )

    assert result is None
    assert model.extract_calls == []


def test_resolve_brand_semantically_skips_without_candidates():
    model = FakeModel()

    result = (
        resolve_brand_semantically(
            model,
            schema="schema",
            analysis=semantic_analysis(
                candidates=[],
            ),
        )
    )

    assert result is None
    assert model.extract_calls == []


def test_resolve_brand_semantically_skips_without_ocr_text():
    model = FakeModel()

    result = (
        resolve_brand_semantically(
            model,
            schema="schema",
            analysis=semantic_analysis(
                candidates=[
                    candidate("JAMESON"),
                ],
                ocr_text="   ",
            ),
        )
    )

    assert result is None
    assert model.extract_calls == []


def test_resolve_brand_semantically_returns_none_without_entities():
    model = FakeModel(
        {
            "entities": {
                "brand_name": [],
            }
        }
    )

    result = (
        resolve_brand_semantically(
            model,
            schema="schema",
            analysis=semantic_analysis(
                candidates=[
                    candidate("JAMESON"),
                ],
                ocr_text="JAMESON",
            ),
        )
    )

    assert result is None

    assert len(
        model.extract_calls
    ) == 1


def test_resolve_brand_semantically_rejects_unobserved_entity():
    model = FakeModel(
        {
            "entities": {
                "brand_name": [
                    "INVENTED BRAND",
                ],
            }
        }
    )

    result = (
        resolve_brand_semantically(
            model,
            schema="schema",
            analysis=semantic_analysis(
                candidates=[
                    candidate("JAMESON"),
                ],
                ocr_text="JAMESON",
            ),
        )
    )

    assert result is None

def test_resolve_brand_semantically_selects_single_candidate():
    model = FakeModel(
        {
            "entities": {
                "brand_name": [
                    "JAMESON",
                ],
            }
        }
    )

    analysis = semantic_analysis(
        candidates=[
            candidate(
                "JAMESON",
                confidence=0.99,
                line_indexes=[1],
            )
        ],
        ocr_text=(
            "JAMESON\n"
            "IRISH WHISKEY"
        ),
    )

    result = (
        resolve_brand_semantically(
            model,
            schema="schema",
            analysis=analysis,
        )
    )

    assert result is not None
    assert result["value"] == "JAMESON"

    assert (
        result["line_indexes"]
        == [1]
    )

    assert result["evidence"] == [
        {
            "line_index": 1,
            "text": "JAMESON",
            "confidence": 0.99,
        }
    ]

    assert result[
        "semantic_observations"
    ] == [
        "JAMESON",
    ]

    assert (
        result["grounding_method"]
        == "grounded_semantic_span"
    )

    assert (
        result["method"]
        == "grounded_semantic_brand_selection"
    )

    assert (
        "semantic_vote_count"
        not in result
    )

    assert (
        "semantic_score"
        not in result
    )


    assert (
        result["method"]
        == "grounded_semantic_brand_selection"
    )

    assert (
        result["grounding_method"]
        == "grounded_semantic_span"
    )


    assert model.extract_calls == [
        {
            "text":
                "JAMESON\n"
                "IRISH WHISKEY",

            "schema":
                "schema",
        }
    ]


def test_resolve_brand_semantically_accepts_exact_confidence_threshold():
    model = FakeModel(
        {
            "entities": {
                "brand_name": [
                    "MALIL",
                ],
            }
        }
    )

    result = (
        resolve_brand_semantically(
            model,
            schema="schema",
            analysis=semantic_analysis(
                candidates=[
                    candidate(
                        "MALIL",
                        confidence=0.80,
                    )
                ],
                ocr_text="MALIL",
            ),
        )
    )

    assert result is not None


def test_two_distinct_grounded_brands_remain_unresolved():
    model = FakeModel(
        {
            "entities": {
                "brand_name": [
                    "ALPHA",
                    "BETA",
                ],
            }
        }
    )

    analysis = semantic_analysis(
        candidates=[],
        ocr_text=(
            "ALPHA\n"
            "BETA"
        ),
    )

    analysis["observations"] = [
        {
            "index": 0,
            "text": "ALPHA",
            "confidence": 0.95,
            "box": None,
        },
        {
            "index": 1,
            "text": "BETA",
            "confidence": 0.95,
            "box": None,
        },
    ]

    result = (
        resolve_brand_semantically(
            model,
            schema="schema",
            analysis=analysis,
        )
    )

    assert result is None


def test_single_multiline_grounded_brand_resolves():
    model = FakeModel(
        {
            "entities": {
                "brand_name": [
                    "ALPHA BETA",
                ],
            }
        }
    )

    analysis = semantic_analysis(
        candidates=[],
        ocr_text=(
            "ALPHA\n"
            "BETA"
        ),
    )

    analysis["observations"] = [
        {
            "index": 0,
            "text": "ALPHA",
            "confidence": 0.95,
            "box": None,
        },
        {
            "index": 1,
            "text": "BETA",
            "confidence": 0.95,
            "box": None,
        },
    ]

    result = (
        resolve_brand_semantically(
            model,
            schema="schema",
            analysis=analysis,
        )
    )

    assert result is not None

    assert (
        identity_key(
            result["value"]
        )
        == "ALPHABETA"
    )

    assert result["line_indexes"] == [
        0,
        1,
    ]

    assert result["evidence"] == [
        {
            "line_index": 0,
            "text": "ALPHA",
            "confidence": 0.95,
        },
        {
            "line_index": 1,
            "text": "BETA",
            "confidence": 0.95,
        },
    ]
