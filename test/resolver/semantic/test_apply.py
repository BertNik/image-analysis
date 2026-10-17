from lib.resolver.semantic.apply import (
    apply_business_relations,
    apply_semantic_brand,
)

from test.resolver.helpers.helpers import (
    candidate,
    semantic_analysis,
)


def test_apply_business_relations_attaches_address():
    analysis = {
        "resolved": {
            "businesses": [
                {
                    "name":
                        "RIVER OAK DISTILLING CO.",

                    "name_method":
                        (
                            "grounded_semantic_"
                            "business_name"
                        ),
                }
            ],
        },
    }

    address_relation = {
        "relation":
            "business_located_at",

        "business_name":
            "RIVER OAK DISTILLING CO.",

        "business_key":
            "RIVEROAKDISTILLINGCO",

        "address":
            "FRANKFORT, KY 40601",

        "business_evidence": [
            {
                "line_index": 8,
                "text":
                    "RIVER OAK DISTILLING CO.",
                "confidence": 0.99,
            }
        ],

        "address_evidence": [
            {
                "line_index": 9,
                "text":
                    "FRANKFORT, KY 40601",
                "confidence": 0.98,
            }
        ],

        "method":
            (
                "grounded_semantic_"
                "business_address_relation"
            ),
    }

    apply_business_relations(
        analysis,
        role_relations=[],
        address_relations=[
            address_relation
        ],
    )

    businesses = (
        analysis["resolved"][
            "businesses"
        ]
    )

    assert len(businesses) == 1

    business = businesses[0]

    assert (
        business["address"]
        ==
        "FRANKFORT, KY 40601"
    )

    assert business[
        "address_evidence"
    ] == [
        {
            "line_index": 9,
            "text":
                "FRANKFORT, KY 40601",
            "confidence": 0.98,
        }
    ]

    assert business[
        "address_relation_evidence"
    ] == [
        address_relation
    ]

def test_apply_business_relations_does_not_choose_conflicting_address():
    analysis = {
        "resolved": {
            "businesses": [
                {
                    "name":
                        "RIVER OAK DISTILLING CO.",
                }
            ],
        },
    }

    first = {
        "business_key":
            "RIVEROAKDISTILLINGCO",
        "business_name":
            "RIVER OAK DISTILLING CO.",
        "address":
            "FRANKFORT, KY 40601",
        "address_evidence":
            [],
        "relation":
            "business_located_at",
    }

    second = {
        "business_key":
            "RIVEROAKDISTILLINGCO",
        "business_name":
            "RIVER OAK DISTILLING CO.",
        "address":
            "LEXINGTON, KY 40507",
        "address_evidence":
            [],
        "relation":
            "business_located_at",
    }

    apply_business_relations(
        analysis,
        role_relations=[],
        address_relations=[
            first,
            second,
        ],
    )

    business = (
        analysis["resolved"][
            "businesses"
        ][0]
    )

    assert (
        business["address"]
        == "FRANKFORT, KY 40601"
    )

    assert (
        len(
            business[
                "address_conflicts"
            ]
        )
        == 1
    )

def test_business_address_relation_attaches_address():
    analysis = {
        "resolved": {
            "businesses": [
                {
                    "name": (
                        "HARBOR STONE "
                        "DISTILLING COMPANY"
                    ),
                    "roles": [
                        "distiller",
                    ],
                },
            ],
        },
    }

    relation = {
        "relation":
            "business_located_at",

        "business_name": (
            "HARBOR STONE "
            "DISTILLING COMPANY"
        ),

        "business_key": (
            "HARBORSTONEDISTILLINGCOMPANY"
        ),

        "address":
            "LOUISVILLE, KENTUCKY 40202",

        "business_evidence": [
            {
                "line_index": 8,
                "text": (
                    "HARBOR STONE "
                    "DISTILLING COMPANY"
                ),
                "confidence": 0.99,
            },
        ],

        "address_evidence": [
            {
                "line_index": 9,
                "text": (
                    "LOUISVILLE, "
                    "KENTUCKY 40202"
                ),
                "confidence": 0.95,
            },
        ],

        "method": (
            "grounded_semantic_"
            "business_address_relation"
        ),
    }

    apply_business_relations(
        analysis,
        role_relations=[],
        address_relations=[
            relation,
        ],
    )

    business = (
        analysis["resolved"]
        ["businesses"][0]
    )

    assert business["address"] == (
        "LOUISVILLE, KENTUCKY 40202"
    )

    assert (
        business[
            "address_relation_evidence"
        ][0]["relation"]
        == "business_located_at"
    )

def test_apply_semantic_brand_updates_analysis():
    analysis = semantic_analysis(
        candidates=[
            candidate(
                "OLD TOM DISTILLERY"
            )
        ]
    )

    semantic_brand = {
        "value":
            "OLD TOM DISTILLERY",

        "line_indexes": [0],

        "ocr_confidence": 0.98,

        "semantic_vote_count": 1,

        "semantic_score": 2.98,

        "method":
            "semantic_span_candidate_selection",
    }

    apply_semantic_brand(
        analysis,
        semantic_brand,
    )

    assert (
        analysis["resolved"][
            "brand_name"
        ]
        is semantic_brand
    )

    assert (
        "brand_name"
        not in analysis[
            "unresolved_required_roles"
        ]
    )

    assert (
        "brand_name"
        not in analysis["unresolved"]
    )

    assert (
        analysis["needs_semantics"]
        is False
    )

    assert (
        analysis["analysis_complete"]
        is True
    )

    assert (
        analysis[
            "deterministic_complete"
        ]
        is False
    )

    assert analysis[
        "semantic_fields"
    ] == [
        "brand_name",
    ]


def test_apply_semantic_brand_preserves_other_unresolved_roles():
    analysis = semantic_analysis(
        candidates=[
            candidate("JAMESON")
        ],
        unresolved_roles=[
            "brand_name",
            "business_name",
        ],
    )

    apply_semantic_brand(
        analysis,
        {
            "value": "JAMESON",
            "line_indexes": [1],
            "method":
                "semantic_span_candidate_selection",
        },
    )

    assert (
        analysis[
            "unresolved_required_roles"
        ]
        == [
            "business_name",
        ]
    )

    assert (
        analysis["needs_semantics"]
        is True
    )

    assert (
        analysis["analysis_complete"]
        is False
    )


def test_apply_semantic_brand_preserves_missing_required_fields():
    analysis = semantic_analysis(
        candidates=[
            candidate("JAMESON")
        ]
    )

    analysis[
        "unresolved_required_fields"
    ] = [
        "alcohol_content",
    ]

    apply_semantic_brand(
        analysis,
        {
            "value": "JAMESON",
            "line_indexes": [1],
            "method":
                "semantic_span_candidate_selection",
        },
    )

    assert (
        analysis["analysis_complete"]
        is False
    )

    assert (
        analysis[
            "unresolved_required_fields"
        ]
        == [
            "alcohol_content",
        ]
    )




def test_apply_semantic_brand_does_not_duplicate_semantic_field():
    analysis = semantic_analysis(
        candidates=[
            candidate("JAMESON")
        ]
    )

    analysis["semantic_fields"] = [
        "brand_name",
    ]

    apply_semantic_brand(
        analysis,
        {
            "value": "JAMESON",
            "line_indexes": [1],
            "method":
                "semantic_span_candidate_selection",
        },
    )

    assert analysis[
        "semantic_fields"
    ] == [
        "brand_name",
    ]
