import pytest
from lib.resolver.semantic.business import (
    resolve_business_addresses_semantically,
    resolve_business_address_relations,
    resolve_business_names_semantically,
    resolve_business_role_relations,
)
from lib.resolver.semantic.grounding import (
    identity_key
)
from test.resolver.helpers.helpers import (
    FakeModel,
)

from lib.resolver.semantic import apply_business_relations

def test_business_address_is_grounded():
    model = FakeModel(
        {
            "entities": {
                "business_address": [
                    "FRANKFORT, KY 40601",
                ],
            }
        }
    )

    analysis = {
        "ocr_text":
            "FRANKFORT, KY 40601",

        "observations": [
            {
                "index": 5,
                "text":
                    "FRANKFORT, KY 40601",
                "confidence": 0.98,
                "box": None,
            }
        ],
    }

    result = (
        resolve_business_addresses_semantically(
            model,
            schema="schema",
            analysis=analysis,
        )
    )

    assert len(result) == 1

    assert (
        result[0]["value"]
        == "FRANKFORT, KY 40601"
    )

    assert result[0]["line_indexes"] == [
        5,
    ]

    assert result[0]["evidence"] == [
        {
            "line_index": 5,
            "text":
                "FRANKFORT, KY 40601",
            "confidence": 0.98,
        }
    ]

    assert (
        result[0]["method"]
        ==
        "grounded_semantic_business_address"
    )

def test_business_address_must_exist_in_ocr():
    model = FakeModel(
        {
            "entities": {
                "business_address": [
                    "NEW YORK, NY 10001",
                ],
            }
        }
    )

    analysis = {
        "ocr_text":
            "FRANKFORT, KY 40601",

        "observations": [
            {
                "index": 0,
                "text":
                    "FRANKFORT, KY 40601",
                "confidence": 0.99,
                "box": None,
            }
        ],
    }

    result = (
        resolve_business_addresses_semantically(
            model,
            schema="schema",
            analysis=analysis,
        )
    )

    assert result == []

def test_multiple_grounded_business_addresses_are_preserved():
    model = FakeModel(
        {
            "entities": {
                "business_address": [
                    "PHILADELPHIA, PA 19103",
                    "NEW YORK, NY 10001",
                ],
            }
        }
    )

    analysis = {
        "ocr_text": (
            "PHILADELPHIA, PA 19103\n"
            "NEW YORK, NY 10001"
        ),

        "observations": [
            {
                "index": 0,
                "text":
                    "PHILADELPHIA, PA 19103",
                "confidence": 0.99,
                "box": None,
            },
            {
                "index": 1,
                "text":
                    "NEW YORK, NY 10001",
                "confidence": 0.99,
                "box": None,
            },
        ],
    }

    result = (
        resolve_business_addresses_semantically(
            model,
            schema="schema",
            analysis=analysis,
        )
    )

    assert {
        item["value"]
        for item in result
    } == {
        "PHILADELPHIA, PA 19103",
        "NEW YORK, NY 10001",
    }

def test_business_address_relation_connects_grounded_endpoints():
    extraction = {
        "relation_extraction": {
            "business_located_at": [
                (
                    "RIVER OAK DISTILLING CO.",
                    "FRANKFORT, KY 40601",
                ),
            ],
        },
    }

    observations = [
        {
            "index": 8,
            "text":
                "RIVER OAK DISTILLING CO.",
            "confidence": 0.99,
            "box": None,
        },
        {
            "index": 9,
            "text":
                "FRANKFORT, KY 40601",
            "confidence": 0.98,
            "box": None,
        },
    ]

    result = (
        resolve_business_address_relations(
            extraction,
            observations,
        )
    )

    assert len(result) == 1

    relation = result[0]

    assert (
        relation["business_name"]
        ==
        "RIVER OAK DISTILLING CO."
    )

    assert (
        relation["business_key"]
        ==
        "RIVEROAKDISTILLINGCO"
    )

    assert (
        relation["address"]
        ==
        "FRANKFORT, KY 40601"
    )

    assert relation[
        "address_evidence"
    ] == [
        {
            "line_index": 9,
            "text":
                "FRANKFORT, KY 40601",
            "confidence": 0.98,
        }
    ]

    assert (
        relation["method"]
        ==
        "grounded_semantic_"
        "business_address_relation"
    )

def test_business_address_relation_rejects_unobserved_address():
    extraction = {
        "relation_extraction": {
            "business_located_at": [
                (
                    "RIVER OAK DISTILLING CO.",
                    "LEXINGTON, KY 40507",
                ),
            ],
        },
    }

    observations = [
        {
            "index": 8,
            "text":
                "RIVER OAK DISTILLING CO.",
            "confidence": 0.99,
            "box": None,
        },
        {
            "index": 9,
            "text":
                "FRANKFORT, KY 40601",
            "confidence": 0.98,
            "box": None,
        },
    ]

    result = (
        resolve_business_address_relations(
            extraction,
            observations,
        )
    )

    assert result == []

def test_business_address_relation_rejects_unobserved_business():
    extraction = {
        "relation_extraction": {
            "business_located_at": [
                (
                    "INVENTED DISTILLERY",
                    "FRANKFORT, KY 40601",
                ),
            ],
        },
    }

    observations = [
        {
            "index": 8,
            "text":
                "RIVER OAK DISTILLING CO.",
            "confidence": 0.99,
            "box": None,
        },
        {
            "index": 9,
            "text":
                "FRANKFORT, KY 40601",
            "confidence": 0.98,
            "box": None,
        },
    ]

    result = (
        resolve_business_address_relations(
            extraction,
            observations,
        )
    )

    assert result == []

def test_distilled_by_relation_resolves_canonical_role():
    extraction = {
        "relation_extraction": {
            "distilled_by": [
                (
                    "PRODUCT",
                    "RIVER OAK DISTILLING CO.",
                ),
            ],
        },
    }

    observations = [
        {
            "index": 4,
            "text":
                "RIVER OAK DISTILLING CO.",
            "confidence": 0.99,
            "box": None,
        }
    ]

    result = (
        resolve_business_role_relations(
            extraction,
            observations,
        )
    )

    assert len(result) == 1

    assert result[0]["relation"] == (
        "distilled_by"
    )

    assert result[0]["role"] == (
        "distiller"
    )

    assert result[0]["source"] == (
        "PRODUCT"
    )

    assert result[0][
        "business_name"
    ] == "RIVER OAK DISTILLING CO."

    assert result[0][
        "business_evidence"
    ] == [
        {
            "line_index": 4,
            "text":
                "RIVER OAK DISTILLING CO.",
            "confidence": 0.99,
        }
    ]

    assert result[0]["method"] == (
        "grounded_semantic_"
        "business_relation"
    )

def test_multiple_business_role_relations_are_preserved():
    extraction = {
        "relation_extraction": {
            "distilled_by": [
                (
                    "PRODUCT",
                    "RIVER OAK DISTILLING CO.",
                ),
            ],
            "bottled_by": [
                (
                    "PRODUCT",
                    "RIVER OAK DISTILLING CO.",
                ),
            ],
        },
    }

    observations = [
        {
            "index": 0,
            "text":
                "RIVER OAK DISTILLING CO.",
            "confidence": 0.99,
            "box": None,
        }
    ]

    result = (
        resolve_business_role_relations(
            extraction,
            observations,
        )
    )

    assert {
        item["role"]
        for item in result
    } == {
        "distiller",
        "bottler",
    }

    assert {
        item["relation"]
        for item in result
    } == {
        "distilled_by",
        "bottled_by",
    }

def test_business_relation_requires_grounded_business():
    extraction = {
        "relation_extraction": {
            "imported_by": [
                (
                    "PRODUCT",
                    "INVENTED IMPORTER",
                ),
            ],
        },
    }

    observations = [
        {
            "index": 0,
            "text":
                "ACTUAL IMPORTER LLC",
            "confidence": 0.99,
            "box": None,
        }
    ]

    result = (
        resolve_business_role_relations(
            extraction,
            observations,
        )
    )

    assert result == []

def test_unknown_business_relation_is_ignored():
    extraction = {
        "relation_extraction": {
            "distributed_by": [
                (
                    "PRODUCT",
                    "ACME DISTRIBUTION LLC",
                ),
            ],
        },
    }

    observations = [
        {
            "index": 0,
            "text":
                "ACME DISTRIBUTION LLC",
            "confidence": 0.99,
            "box": None,
        }
    ]

    result = (
        resolve_business_role_relations(
            extraction,
            observations,
        )
    )

    assert result == []

def test_multilingual_relation_uses_canonical_role():
    extraction = {
        "relation_extraction": {
            "distilled_by": [
                (
                    "PRODUCTO",
                    "DESTILERIA RIO CLARO",
                ),
            ],
        },
    }

    observations = [
        {
            "index": 2,
            "text":
                "DESTILERIA RIO CLARO",
            "confidence": 0.98,
            "box": None,
        }
    ]

    result = (
        resolve_business_role_relations(
            extraction,
            observations,
        )
    )

    assert len(result) == 1

    assert result[0]["role"] == (
        "distiller"
    )

    assert result[0][
        "business_name"
    ] == "DESTILERIA RIO CLARO"
def test_business_name_is_grounded():
    model = FakeModel(
        {
            "entities": {
                "business_name": [
                    (
                        "RIVER OAK "
                        "DISTILLING CO."
                    ),
                ],
            }
        }
    )

    analysis = {
        "ocr_text": (
            "RIVER OAK "
            "DISTILLING CO."
        ),

        "observations": [
            {
                "index": 0,
                "text": (
                    "RIVER OAK "
                    "DISTILLING CO."
                ),
                "confidence": 0.99,
                "box": None,
            }
        ],
    }

    result = (
        resolve_business_names_semantically(
            model,
            schema="schema",
            analysis=analysis,
        )
    )

    assert len(result) == 1

    assert (
        result[0]["name"]
        ==
        "RIVER OAK DISTILLING CO."
    )

    assert (
        result[0]["line_indexes"]
        == [0]
    )

    assert result[0]["evidence"] == [
        {
            "line_index": 0,
            "text":
                "RIVER OAK DISTILLING CO.",
            "confidence": 0.99,
        }
    ]

def test_business_name_must_exist_in_ocr():
    model = FakeModel(
        {
            "entities": {
                "business_name": [
                    "INVENTED COMPANY",
                ],
            }
        }
    )

    analysis = {
        "ocr_text":
            "RIVER OAK DISTILLING CO.",

        "observations": [
            {
                "index": 0,
                "text":
                    "RIVER OAK DISTILLING CO.",
                "confidence": 0.99,
                "box": None,
            }
        ],
    }

    result = (
        resolve_business_names_semantically(
            model,
            schema="schema",
            analysis=analysis,
        )
    )

    assert result == []

def test_multiline_business_name_is_grounded():
    model = FakeModel(
        {
            "entities": {
                "business_name": [
                    (
                        "RIVER OAK "
                        "DISTILLING CO."
                    ),
                ],
            }
        }
    )

    analysis = {
        "ocr_text": (
            "RIVER OAK\n"
            "DISTILLING CO."
        ),

        "observations": [
            {
                "index": 3,
                "text": "RIVER OAK",
                "confidence": 0.99,
                "box": None,
            },
            {
                "index": 4,
                "text": "DISTILLING CO.",
                "confidence": 0.98,
                "box": None,
            },
        ],
    }

    result = (
        resolve_business_names_semantically(
            model,
            schema="schema",
            analysis=analysis,
        )
    )

    assert len(result) == 1

    assert (
        identity_key(
            result[0]["name"]
        )
        ==
        "RIVEROAKDISTILLINGCO"
    )

    assert (
        result[0]["line_indexes"]
        == [3, 4]
    )

def test_multiple_grounded_business_names_are_preserved():
    model = FakeModel(
        {
            "entities": {
                "business_name": [
                    "ALPHA IMPORTS INC.",
                    "BETA SPIRITS LLC",
                ],
            }
        }
    )

    analysis = {
        "ocr_text": (
            "ALPHA IMPORTS INC.\n"
            "BETA SPIRITS LLC"
        ),

        "observations": [
            {
                "index": 0,
                "text":
                    "ALPHA IMPORTS INC.",
                "confidence": 0.99,
                "box": None,
            },
            {
                "index": 1,
                "text":
                    "BETA SPIRITS LLC",
                "confidence": 0.99,
                "box": None,
            },
        ],
    }

    result = (
        resolve_business_names_semantically(
            model,
            schema="schema",
            analysis=analysis,
        )
    )

    assert {
        item["name"]
        for item in result
    } == {
        "ALPHA IMPORTS INC.",
        "BETA SPIRITS LLC",
    }

def test_semantic_relations_create_business_without_legacy_parser():
    analysis = {
        "resolved": {
            "businesses": [],
        },
    }

    role_relation = {
        "relation":
            "distilled_by",

        "role":
            "distiller",

        "source":
            "RIVER OAK DISTILLING CO.",

        "business_name":
            "RIVER OAK DISTILLING CO.",

        "business_key":
            "RIVEROAKDISTILLINGCO",

        "business_evidence": [
            {
                "line_index": 8,
                "text":
                    "RIVER OAK DISTILLING CO.",
                "confidence": 0.99,
            }
        ],

        "business_line_indexes": [
            8,
        ],

        "grounding_method":
            "grounded_semantic_span",

        "method":
            "grounded_semantic_business_relation",
    }

    apply_business_relations(
        analysis,
        role_relations=[
            role_relation,
        ],
        address_relations=[],
    )

    businesses = (
        analysis["resolved"][
            "businesses"
        ]
    )

    assert len(businesses) == 1

    business = businesses[0]

    assert (
        business["name"]
        ==
        "RIVER OAK DISTILLING CO."
    )

    assert business["roles"] == [
        "distiller",
    ]

    assert business[
        "role_evidence"
    ] == [
        role_relation,
    ]
