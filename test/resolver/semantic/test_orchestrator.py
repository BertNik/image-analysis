from lib.resolver.semantic.orchestrator import resolve_semantic_roles
from lib.resolver.semantic import orchestrator 

from test.resolver.helpers.helpers import (
    FakeModel,
)
import lib.resolver.semantic as semantic

def test_semantic_coordinator_uses_one_model_extraction():
    model = FakeModel(
        {
            "entities": {
                "brand_name": [
                    "JAMESON",
                ],
                "class_type": [
                    "IRISH WHISKEY",
                ],
                "business_name": [
                    "JOHN JAMESON & SON",
                ],
            }
        }
    )

    analysis = {
        "ocr_text": (
            "JAMESON\n"
            "IRISH WHISKEY\n"
            "JOHN JAMESON & SON"
        ),

        "observations": [
            {
                "index": 0,
                "text": "JAMESON",
                "confidence": 0.99,
                "box": None,
            },
            {
                "index": 1,
                "text": "IRISH WHISKEY",
                "confidence": 0.99,
                "box": None,
            },
            {
                "index": 2,
                "text":
                    "JOHN JAMESON & SON",
                "confidence": 0.99,
                "box": None,
            },
        ],

        "resolved": {
            "brand_name": None,
            "class_type": None,
            "businesses": [],
        },

        "unresolved_required_fields":
            [],

        "unresolved_required_roles": [
            "brand_name",
            "class_type",
        ],

        "unresolved": {
            "brand_name": {},
            "class_type": {},
        },

        "semantic_fields": [],

        "needs_semantics": True,

        "analysis_complete": False,

        "deterministic_complete": False,
    }

    result, _ = (
        resolve_semantic_roles(
            model,
            schema="schema",
            analysis=analysis,
        )
    )

    # Critical regression check.
    assert len(
        model.extract_calls
    ) == 1

    assert model.extract_calls == [
        {
            "text": (
                "JAMESON\n"
                "IRISH WHISKEY\n"
                "JOHN JAMESON & SON"
            ),
            "schema": "schema",
        }
    ]

    assert (
        result[
            "brand_name"
        ]["value"]
        == "JAMESON"
    )

    assert (
        result[
            "class_type"
        ]["value"]
        == "IRISH WHISKEY"
    )

    assert (
        result[
            "business_names"
        ][0]["name"]
        == "JOHN JAMESON & SON"
    )

    businesses = analysis[
        "resolved"
    ]["businesses"]

    assert len(businesses) == 1

    assert (
        businesses[0]["name"]
        == "JOHN JAMESON & SON"
    )

    assert (
        businesses[0]["name_method"]
        ==
        "grounded_semantic_business_name"
    )

    assert businesses[0][
        "name_evidence"
    ] == [
        {
            "line_index": 2,
            "text":
                "JOHN JAMESON & SON",
            "confidence": 0.99,
        }
    ]

    assert set(
        analysis[
            "semantic_fields"
        ]
    ) == {
        "brand_name",
        "class_type",
        "business_name",
    }


def test_resolve_semantic_roles_leaves_declined_roles_unresolved(
    monkeypatch,
):
    analysis = {
        "resolved": {
            "brand_name": None,
            "class_type": None,
        },
        "unresolved_required_fields": [],
        "unresolved_required_roles": [
            "brand_name",
            "class_type",
        ],
        "unresolved": {
            "brand_name": {},
            "class_type": {},
        },
        "semantic_fields": [],
    }

    monkeypatch.setattr(
        semantic,
        "resolve_brand_semantically",
        lambda *args, **kwargs:
            None,
    )

    monkeypatch.setattr(
        orchestrator,
        "resolve_class_type_semantically",
        lambda *args, **kwargs:
            None,
    )
    
    results, extraction = (
        resolve_semantic_roles(
            model="model",
            schema="schema",
            analysis=analysis,
        )
    )

    assert results == {}
    assert extraction is None

    assert analysis[
        "unresolved_required_roles"
    ] == [
        "brand_name",
        "class_type",
    ]

    assert analysis[
        "semantic_fields"
    ] == []

def test_semantic_business_name_creates_business_record():
    analysis = {
        "ocr_text":
            "ACME SPIRITS LLC",

        "observations": [
            {
                "index": 4,
                "text":
                    "ACME SPIRITS LLC",
                "confidence":
                    0.98,
                "box":
                    None,
            }
        ],

        "resolved": {
            "brand_name":
                None,

            "class_type":
                None,

            "businesses":
                [],
        },

        "unresolved_required_fields":
            [],

        "unresolved_required_roles":
            [],

        "unresolved":
            {},

        "semantic_fields":
            [],

        "needs_semantics":
            True,
    }

    model = FakeModel(
        {
            "entities": {
                "business_name": [
                    "ACME SPIRITS LLC",
                ],
            }
        }
    )

    resolve_semantic_roles(
        model,
        schema="schema",
        analysis=analysis,
    )

    assert (
        analysis["resolved"][
            "businesses"
        ]
        == [
            {
                "name":
                    "ACME SPIRITS LLC",

                "observed_text":
                    "ACME SPIRITS LLC",

                "line_indexes":
                    [4],

                "name_evidence": [
                    {
                        "line_index":
                            4,

                        "text":
                            "ACME SPIRITS LLC",

                        "confidence":
                            0.98,
                    }
                ],

                "semantic_observations":
                    [
                        "ACME SPIRITS LLC"
                    ],

                "name_grounding_method":
                    "grounded_semantic_span",

                "name_method":
                    (
                        "grounded_semantic_"
                        "business_name"
                    ),
            }
        ]
    )


    
def test_semantic_business_name_enriches_resolved_business(
    monkeypatch,
):
    analysis = {
        "ocr_text":
            (
                "RIVER OAK "
                "DISTILLING CO."
            ),

        "observations": [
            {
                "index": 0,
                "text":
                    (
                        "RIVER OAK "
                        "DISTILLING CO."
                    ),
                "confidence":
                    0.99,
                "box":
                    None,
            }
        ],

        "resolved": {
            "brand_name":
                None,

            "class_type":
                None,

            "businesses": [
                {
                    "name":
                        (
                            "RIVER OAK "
                            "DISTILLING CO."
                        ),

                    "roles": [
                        "distiller",
                        "bottler",
                    ],

                    "address":
                        "FRANKFORT, KY 40601",
                }
            ],
        },

        "unresolved_required_fields":
            [],

        "unresolved_required_roles":
            [],

        "unresolved":
            {},

        "semantic_fields":
            [],

        "needs_semantics":
            True,
    }

    extraction = {
        "entities": {
            "business_name": [
                (
                    "RIVER OAK "
                    "DISTILLING CO."
                ),
            ],
        }
    }

    model = FakeModel(
        extraction
    )

    result, _ = (
        resolve_semantic_roles(
            model,
            schema="schema",
            analysis=analysis,
        )
    )

    businesses = (
        analysis["resolved"][
            "businesses"
        ]
    )

    assert len(businesses) == 1

    business = businesses[0]

    # Semantic ownership of name.
    assert (
        business["name"]
        ==
        "RIVER OAK DISTILLING CO."
    )

    assert (
        business["name_method"]
        ==
        "grounded_semantic_business_name"
    )

    assert business[
        "name_evidence"
    ] == [
        {
            "line_index": 0,
            "text":
                "RIVER OAK DISTILLING CO.",
            "confidence":
                0.99,
        }
    ]

    # Existing role/address behavior survives
    # this narrow ownership migration.
    assert set(
        business["roles"]
    ) == {
        "distiller",
        "bottler",
    }

    assert (
        business["address"]
        ==
        "FRANKFORT, KY 40601"
    )

    assert (
        "business_name"
        in analysis[
            "semantic_fields"
        ]
    )

    assert (
        "business_names"
        in result
    )



    




    