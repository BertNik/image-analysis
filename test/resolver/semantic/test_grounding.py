from lib.resolver.semantic.grounding import (
    collect_grounded_entities,
)

def test_collect_grounded_entities_deduplicates_identity():
    extraction = {
        "entities": {
            "business_name": [
                "RIVER OAK DISTILLING CO.",
                "River-Oak Distilling Co",
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
        collect_grounded_entities(
            extraction,
            "business_name",
            observations,
        )
    )

    assert len(result) == 1

    assert result[0][
        "grounded"
    ]["value"] == (
        "RIVER OAK DISTILLING CO."
    )

    assert result[0][
        "semantic_observations"
    ] == [
        "RIVER OAK DISTILLING CO.",
        "River-Oak Distilling Co",
    ]

def test_collect_grounded_entities_rejects_unobserved_value():
    extraction = {
        "entities": {
            "business_name": [
                "INVENTED COMPANY",
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
        collect_grounded_entities(
            extraction,
            "business_name",
            observations,
        )
    )

    assert result == []

