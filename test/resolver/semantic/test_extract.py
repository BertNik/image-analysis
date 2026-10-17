import pytest

from lib.resolver.semantic.extraction import (
    ensure_extraction,
    get_entity_values,
    get_relation_pairs,
    normalize_relation_pair,
)

from test.resolver.helpers import (
    helpers,
)

@pytest.mark.parametrize(
    "value",
    [
        {
            "source": "SOURCE",
        },
        {
            "target": "TARGET",
        },
        {},
        {
            "source": "",
            "target": "TARGET",
        },
        {
            "source": "SOURCE",
            "target": "",
        },
    ],
)
def test_normalize_relation_pair_rejects_incomplete_dict(
    value,
):
    assert (
        normalize_relation_pair(
            value
        )
        is None
    )

def test_normalize_relation_pair_from_source_target_dict():
    value = {
        "source": "BOURBON WHISKEY",
        "target": "HARBOR DISTILLING CO.",
    }

    result = normalize_relation_pair(
        value
    )

    assert result == (
        "BOURBON WHISKEY",
        "HARBOR DISTILLING CO.",
    )


@pytest.mark.parametrize(
    "value",
    [
        "not a relation",
        123,
        3.14,
        True,
        object(),
    ],
)

def test_normalize_relation_pair_rejects_unsupported_types(
    value,
):
    assert (
        normalize_relation_pair(
            value
        )
        is None
    )

@pytest.mark.parametrize(
    (
        "value",
        "expected",
    ),
    [
        (
            {
                "source": "SOURCE A",
                "target": "TARGET A",
            },
            (
                "SOURCE A",
                "TARGET A",
            ),
        ),
        (
            {
                "head": "SOURCE B",
                "tail": "TARGET B",
            },
            (
                "SOURCE B",
                "TARGET B",
            ),
        ),
        (
            {
                "subject": "SOURCE C",
                "object": "TARGET C",
            },
            (
                "SOURCE C",
                "TARGET C",
            ),
        ),
        (
            {
                "from": "SOURCE D",
                "to": "TARGET D",
            },
            (
                "SOURCE D",
                "TARGET D",
            ),
        ),
    ],
)
    
def test_normalize_relation_pair_dict_shapes(
    value,
    expected,
):
    assert normalize_relation_pair(
        value
    ) == expected

def test_get_relation_pairs_accepts_described_relation_key():
    extraction = {
        "relation_extraction": {
            (
                "distilled_by: "
                "The target is the business "
                "responsible for distilling "
                "the product."
            ): [
                (
                    "PRODUCT",
                    "RIVER OAK DISTILLING CO.",
                ),
            ],

            "distilled_by": [],
        },
    }

    result = (
        get_relation_pairs(
            extraction,
            "distilled_by",
        )
    )

    assert result == [
        (
            "PRODUCT",
            "RIVER OAK DISTILLING CO.",
        )
    ]

@pytest.mark.parametrize(
    "extraction, expected",
    [
        (
            {},
            [],
        ),
        (
            {
                "entities": {},
            },
            [],
        ),
        (
            {
                "entities": {
                    "brand_name": None,
                },
            },
            [],
        ),
        (
            {
                "entities": {
                    "brand_name":
                        "MALIBU",
                },
            },
            [
                "MALIBU",
            ],
        ),
        (
            {
                "entities": {
                    "brand_name": [
                        "MALIBU",
                        "CAPTAIN MORGAN",
                    ],
                },
            },
            [
                "MALIBU",
                "CAPTAIN MORGAN",
            ],
        ),
        (
            {
                "entities": {
                    "brand_name": [
                        {
                            "text":
                                "MALIBU",
                        },
                        {
                            "value":
                                "JAMESON",
                        },
                        {
                            "span":
                                "BACARDI",
                        },
                    ],
                },
            },
            [
                "MALIBU",
                "JAMESON",
                "BACARDI",
            ],
        ),
        (
            {
                "entities": {
                    "brand_name": [
                        "  MALIBU  ",
                        "",
                        {
                            "text": " ",
                        },
                    ],
                },
            },
            [
                "MALIBU",
            ],
        ),
        (
            {
                "entities": {
                    "brand_name": [
                        123,
                    ],
                },
            },
            [
                "123",
            ],
        ),
    ],
)
def test_get_entity_values(
    extraction,
    expected,
):
    result = (
        get_entity_values(
            extraction,
            "brand_name",
        )
    )

    assert result == expected


def test_get_entity_values_uses_requested_label():
    extraction = {
        "entities": {
            "brand_name": [
                "JAMESON",
            ],
            "business_name": [
                "JOHN JAMESON & SON",
            ],
        }
    }

    assert (
        get_entity_values(
            extraction,
            "business_name",
        )
        == [
            "JOHN JAMESON & SON",
        ]
    )

def test_ensure_extraction_uses_supplied_extraction():
    supplied = {
        "entities": {},
    }

    model = helpers.FakeModel()

    result = (
        ensure_extraction(
            model,
            "schema",
            {
                "ocr_text": "LABEL",
                "observations": [
                    {
                        "index": 0,
                        "text": "LABEL",
                    }
                ],
            },
            supplied,
        )
    )

    assert result is supplied
    assert model.extract_calls == []