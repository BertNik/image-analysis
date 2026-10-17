import numpy as np
import pytest
import lib.analyze_image as analyze_image_module


def usable_route(
    status,
    reasons=None,
):
    return {
        "status": status,
        "reasons": (
            reasons
            if reasons is not None
            else []
        ),
        "usability": {
            "usable": True,
            "usable_count": 1,
            "usable_observations": [],
        },
    }

def test_analyze_image_uses_warning_heading_box(
    monkeypatch,
):
    image = np.zeros(
        (200, 400, 3),
        dtype=np.uint8,
    )

    warning = {
        "observed": True,
        "complete": True,
        "text_exact": True,
        "heading_uppercase": True,
        "heading_bold": None,

        "heading_box": [
            10,
            20,
            200,
            50,
        ],

        "line_indexes": [
            1,
            2,
            3,
        ],

        "evidence": [],
    }

    analysis = {
        "government_warning":
            warning,

        "observations": [
            {
                "index": 1,
                "text":
                    "GOVERNMENT WARNING:",
                "box": [
                    10,
                    20,
                    200,
                    50,
                ],
            },
        ],

        "needs_semantics":
            False,
    }

    monkeypatch.setattr(
        "lib.analyze_image.ocr_function",
        lambda image: {
            "text": "OCR"
        },
    )

    monkeypatch.setattr(
        (
            "lib.analyze_image."
            "analyze_label_deterministically"
        ),
        lambda *args, **kwargs:
            analysis,
    )

    visual_calls = []

    def fake_visual_resolver(
        *,
        image,
        heading_box,
        observations,
        warning_line_indexes,
    ):
        visual_calls.append(
            {
                "heading_box":
                    heading_box,

                "observations":
                    observations,

                "warning_line_indexes":
                    warning_line_indexes,
            }
        )

        return {
            "value": True,
            "method":
                "relative_warning_stroke_width",
        }

    monkeypatch.setattr(
        (
            "lib.analyze_image."
            "analyze_heading_boldness"
        ),
        fake_visual_resolver,
    )

    monkeypatch.setattr(
        "lib.analyze_image.route_analysis",
        lambda analysis: {
            "status": "RESOLVED",
            "reasons": [],
            "usability": {
                "usable": True,
            },
        },
    )

    result = analyze_image_module.analyze_image(
        image,
    )

    assert len(visual_calls) == 1

    assert visual_calls[0][
        "heading_box"
    ] == [
        10,
        20,
        200,
        50,
    ]

    assert visual_calls[0][
        "warning_line_indexes"
    ] == [
        1,
        2,
        3,
    ]

    assert (
        result["analysis"]
        ["government_warning"]
        ["heading_bold"]
        is True
    )

def test_analyze_image_recovers_heading_box_from_evidence(
    monkeypatch,
):
    image = np.zeros(
        (200, 400, 3),
        dtype=np.uint8,
    )

    analysis = {
        "government_warning": {
            "observed": True,
            "complete": True,
            "text_exact": True,
            "heading_uppercase": True,
            "heading_bold": None,

            # Deliberately absent.
            "heading_box": None,

            "line_indexes": [
                5,
                6,
            ],

            "evidence": [
                {
                    "line_index": 4,
                    "text":
                        "SOMETHING ELSE",
                    "box": [
                        1,
                        1,
                        20,
                        20,
                    ],
                },
                {
                    "line_index": 5,
                    "text":
                        "GOVERNMENT WARNING:",
                    "box": [
                        50,
                        60,
                        250,
                        90,
                    ],
                },
            ],
        },

        "observations": [],

        "needs_semantics":
            False,
    }

    monkeypatch.setattr(
        "lib.analyze_image.ocr_function",
        lambda image: {
            "text": "OCR"
        },
    )

    monkeypatch.setattr(
        (
            "lib.analyze_image."
            "analyze_label_deterministically"
        ),
        lambda *args, **kwargs:
            analysis,
    )

    captured = {}

    def fake_visual_resolver(
        *,
        image,
        heading_box,
        observations,
        warning_line_indexes,
    ):
        captured[
            "heading_box"
        ] = heading_box

        return {
            "value": True,
            "method":
                "relative_warning_stroke_width",
        }

    monkeypatch.setattr(
        (
            "lib.analyze_image."
            "analyze_heading_boldness"
        ),
        fake_visual_resolver,
    )

    monkeypatch.setattr(
        "lib.analyze_image.route_analysis",
        lambda analysis: {
            "status": "RESOLVED",
            "reasons": [],
            "usability": {
                "usable": True,
            },
        },
    )

    result = analyze_image_module.analyze_image(
        image,
    )

    assert captured[
        "heading_box"
    ] == [
        50,
        60,
        250,
        90,
    ]

    assert (
        result["analysis"]
        ["government_warning"]
        ["heading_bold"]
        is True
    )

def test_analyze_image_verifies_when_semantic_model_unavailable(
    monkeypatch,
):
    image = np.zeros(
        (100, 200, 3),
        dtype=np.uint8,
    )

    analysis = {
        "ocr_text":
            "HARBOR STONE",

        "observations": [
            {
                "index": 0,
                "text":
                    "HARBOR STONE",

                "confidence":
                    0.99,

                "box": [
                    10,
                    10,
                    150,
                    30,
                ],
            },
        ],

        "resolved": {
            "brand_name":
                None,
        },

        "unresolved_required_fields":
            [],

        "unresolved_required_roles": [
            "brand_name",
        ],

        "needs_semantics":
            True,

        "government_warning":
            None,
    }

    usability = {
        "usable": True,
    }

    monkeypatch.setattr(
        analyze_image_module,
        "analyze_label_deterministically",
        lambda *args, **kwargs:
            analysis,
    )

    monkeypatch.setattr(
        analyze_image_module,
        "route_analysis",
        lambda analysis: {
            "status":
                "SEMANTIC_FALLBACK",

            "reasons": [
                "semantic_resolution_required",
            ],

            "usability":
                usability,
        },
    )

    def semantic_should_not_run(
        *args,
        **kwargs,
    ):
        pytest.fail(
            "Semantic resolution should not run "
            "without semantic dependencies."
        )

    monkeypatch.setattr(
        analyze_image_module,
        "resolve_semantic_roles",
        semantic_should_not_run,
    )

    result = (
        analyze_image_module
        .analyze_image(
            image=image,

            semantic_model=None,
            semantic_schema=None,
        )
    )

    assert result["status"] == (
        "VERIFY"
    )

    assert result["reasons"] == [
        "semantic_model_unavailable",
    ]

    assert result[
        "usability"
    ] == usability
    
def test_analyze_image_marks_boldness_unresolved_without_heading_box(
    monkeypatch,
):
    image = np.zeros(
        (200, 400, 3),
        dtype=np.uint8,
    )

    analysis = {
        "government_warning": {
            "observed": True,
            "complete": True,
            "text_exact": True,
            "heading_uppercase": True,
            "heading_bold": None,

            "heading_box":
                None,

            "line_indexes": [
                5,
                6,
            ],

            "evidence": [
                {
                    "line_index": 5,
                    "text":
                        "NOT THE HEADING",
                    "box": None,
                },
            ],
        },

        "observations": [],

        "needs_semantics":
            False,
    }

    monkeypatch.setattr(
        "lib.analyze_image.ocr_function",
        lambda image: {
            "text": "OCR"
        },
    )

    monkeypatch.setattr(
        (
            "lib.analyze_image."
            "analyze_label_deterministically"
        ),
        lambda *args, **kwargs:
            analysis,
    )

    def visual_should_not_run(
        **kwargs,
    ):
        pytest.fail(
            "Visual resolver should not run "
            "without a heading box."
        )

    monkeypatch.setattr(
        (
            "lib.analyze_image."
            "analyze_heading_boldness"
        ),
        visual_should_not_run,
    )

    monkeypatch.setattr(
        "lib.analyze_image.route_analysis",
        lambda analysis: {
            "status": "VERIFY",
            "reasons": [],
            "usability": {
                "usable": True,
            },
        },
    )

    result = analyze_image_module.analyze_image(
        image,
    )

    warning = (
        result["analysis"]
        ["government_warning"]
    )

    assert (
        warning["heading_bold"]
        is None
    )

    assert (
        warning["boldness_evidence"]
        ["value"]
        is None
    )

    assert (
        warning["boldness_evidence"]
        ["reason"]
        ==
        "warning_heading_box_unavailable"
    )

def test_analyze_image_reports_unresolved_semantic_roles(
    monkeypatch,
):
    analysis = {
        "ocr_text":
            "HARBOR STONE\n"
            "BOURBON WHISKEY",

        "observations": [
            {
                "index": 0,
                "text": "HARBOR STONE",
                "confidence": 0.99,
                "box": [10, 10, 100, 30],
            },
            {
                "index": 1,
                "text": "BOURBON WHISKEY",
                "confidence": 0.99,
                "box": [10, 40, 140, 60],
            },
        ],

        "resolved": {
            "brand_name": None,
            "class_type": None,
        },

        "unresolved_required_fields": [],

        "unresolved_required_roles": [
            "brand_name",
            "class_type",
        ],

        "needs_semantics": True,

        "government_warning": None,
    }

    usability = {
        "usable": True,
    }

    # Patch these names where analyze_image.py
    # actually imports/uses them.
    monkeypatch.setattr(
        analyze_image_module,
        "analyze_label_deterministically",
        lambda *args, **kwargs:
            analysis,
    )

    monkeypatch.setattr(
        analyze_image_module,
        "route_analysis",
        lambda analysis: {
            "status":
                "SEMANTIC_FALLBACK",

            "reasons": [],

            "usability":
                usability,
        },
    )

    # Semantic resolution executes, but deliberately
    # leaves both required roles unresolved.
    extraction = {
        "entities": {},
    }

    monkeypatch.setattr(
        analyze_image_module,
        "resolve_semantic_roles",
        lambda *args, **kwargs: (
            {},
            extraction,
        ),
    )

    result = (
        analyze_image_module
        .analyze_image(
            image=np.zeros(
                (100, 200, 3),
                dtype=np.uint8,
            ),
            semantic_model=object(),
            semantic_schema=object(),
        )
    )

    assert result["status"] == "VERIFY"

    assert result["reasons"] == [
        "brand_name_semantic_unresolved",
        "class_type_semantic_unresolved",
    ]

    assert result["usability"] == {
        "usable": True,
    }

def test_analyze_image_runs_semantics_when_deterministic_field_is_missing(
    monkeypatch,
):
    image = np.zeros(
        (20, 20, 3),
        dtype=np.uint8,
    )

    analysis = {
        "observations": [
            {
                "index": 0,
                "text": "HORNITOS",
                "confidence": 0.99,
                "box": None,
            }
        ],
        "resolved": {
            "brand_name": None,
            "class_type": None,
            "alcohol_content": None,
            "proof": None,
            "net_contents": None,
            "businesses": [],
        },
        "unresolved_required_fields": [
            "alcohol_content",
        ],
        "unresolved_required_roles": [
            "brand_name",
            "class_type",
        ],
        "unresolved": {
            "alcohol_content": {
                "reason":
                    "alcohol_content_not_observed",
            },
            "brand_name": {
                "reason":
                    "brand_requires_semantic_resolution",
            },
            "class_type": {
                "reason":
                    "class_type_requires_semantic_resolution",
            },
        },
        "not_observed_fields": [
            "net_contents",
        ],
        "needs_semantics": True,
        "analysis_complete": False,
        "deterministic_complete": False,
        "semantic_fields": [],
    }

    monkeypatch.setattr(
        analyze_image_module,
        "ocr_function",
        lambda image: {
            "text": "HORNITOS",
        },
    )

    monkeypatch.setattr(
        analyze_image_module,
        "analyze_label_deterministically",
        lambda *args, **kwargs:
            analysis,
    )

    # The missing deterministic field makes both the initial
    # and final disposition VERIFY.
    monkeypatch.setattr(
        analyze_image_module,
        "route_analysis",
        lambda analysis:
            usable_route(
                "VERIFY",
                [
                    "alcohol_content_not_resolved",
                ],
            ),
    )

    called = False

    brand = {
        "value": "HORNITOS",
        "observed_text": "HORNITOS",
        "line_indexes": [0],
        "evidence": [
            {
                "line_index": 0,
                "text": "HORNITOS",
                "confidence": 0.99,
            }
        ],
        "semantic_observations": [
            "HORNITOS",
        ],
        "grounding_method":
            "grounded_semantic_span",
        "method":
            "grounded_semantic_brand_selection",
    }

    def fake_resolve_semantic_roles(
        model,
        schema,
        received_analysis,
    ):
        nonlocal called
        called = True

        assert model == "model"
        assert schema == "schema"
        assert received_analysis is analysis

        received_analysis["resolved"][
            "brand_name"
        ] = brand

        received_analysis[
            "unresolved_required_roles"
        ].remove(
            "brand_name"
        )

        received_analysis[
            "unresolved"
        ].pop(
            "brand_name"
        )

        received_analysis[
            "semantic_fields"
        ].append(
            "brand_name"
        )

        extraction = {
            "entities": {
                "brand_name": [
                    "HORNITOS",
                ],
            },
        }

        return (
            {
                "brand_name": brand,
            },
            extraction,
        )

    monkeypatch.setattr(
        analyze_image_module,
        "resolve_semantic_roles",
        fake_resolve_semantic_roles,
    )

    result = (
        analyze_image_module
        .analyze_image(
            image,
            semantic_model="model",
            semantic_schema="schema",
            brand_required=True,
        )
    )

    assert called is True

    assert (
        result["status"]
        == "VERIFY"
    )

    assert result["reasons"] == [
        "alcohol_content_not_resolved",
    ]

    assert result[
        "semantic_result"
    ] == {
        "brand_name": brand,
    }

    assert result["analysis"][
        "resolved"
    ]["brand_name"] == brand

    assert result["timing"][
        "semantic_seconds"
    ] >= 0.0

def test_analyze_image_resolved_without_semantic_fallback(
    monkeypatch,
):
    image = np.zeros(
        (20, 20, 3),
        dtype=np.uint8,
    )

    analysis = {
        "observations": [
            {
                "index": 0,
                "text": "750 mL",
                "confidence": 0.99,
                "box": None,
            }
        ],

        "resolved": {
            "brand_name": None,
            "class_type": {
                "value": "RUM",
            },
            "net_contents": {
                "value": 750,
                "unit": "mL",
            },
        },

        "unresolved_required_fields": [],
        "unresolved_required_roles": [],
        "semantic_fields": [],
    }

    monkeypatch.setattr(
        analyze_image_module,
        "ocr_function",
        lambda image: {
            "text": "RUM\n750 mL",
            "lines": [
                "RUM",
                "750 mL",
            ],
            "scores": [
                0.99,
                0.99,
            ],
            "boxes": [],
        },
    )

    monkeypatch.setattr(
        analyze_image_module,
        "analyze_label_deterministically",
        lambda *args, **kwargs:
            analysis,
    )

    monkeypatch.setattr(
        analyze_image_module,
        "route_analysis",
        lambda analysis:
            usable_route(
                "RESOLVED"
            ),
    )

    semantic_called = False

    def fake_resolve_semantic_roles(
        *args,
        **kwargs,
    ):
        nonlocal semantic_called
        semantic_called = True

        raise AssertionError(
            "Semantic coordinator "
            "should not run"
        )

    monkeypatch.setattr(
        analyze_image_module,
        "resolve_semantic_roles",
        fake_resolve_semantic_roles,
    )

    result = (
        analyze_image_module
        .analyze_image(
            image,
            semantic_model="model",
            semantic_schema="schema",
        )
    )

    assert (
        result["status"]
        == "RESOLVED"
    )

    assert result["reasons"] == []

    assert (
        result["semantic_result"]
        is None
    )

    assert (
        result["timing"][
            "semantic_seconds"
        ]
        == 0.0
    )

    assert semantic_called is False


def test_analyze_image_semantic_coordinator_resolves_all_roles(
    monkeypatch,
):
    image = np.zeros(
        (20, 20, 3),
        dtype=np.uint8,
    )

    analysis = {
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
                "confidence": 0.98,
                "box": None,
            },
        ],
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
        "needs_semantics": True,
        "analysis_complete": False,
        "deterministic_complete": False,
        "semantic_fields": [],
    }

    monkeypatch.setattr(
        analyze_image_module,
        "ocr_function",
        lambda image: {
            "text": (
                "JAMESON\n"
                "IRISH WHISKEY"
            ),
        },
    )

    monkeypatch.setattr(
        analyze_image_module,
        "analyze_label_deterministically",
        lambda *args, **kwargs:
            analysis,
    )

    routes = iter(
        [
            usable_route(
                "SEMANTIC_FALLBACK"
            ),
            usable_route(
                "RESOLVED"
            ),
        ]
    )

    monkeypatch.setattr(
        analyze_image_module,
        "route_analysis",
        lambda analysis:
            next(routes),
    )

    brand = {
        "value": "JAMESON",
        "method":
            "grounded_semantic_brand_selection",
    }

    class_type = {
        "value": "IRISH WHISKEY",
        "method":
            "grounded_semantic_class_type_selection",
    }

    def fake_resolve_semantic_roles(
        model,
        schema,
        received_analysis,
    ):
        assert model == "model"
        assert schema == "schema"
        assert received_analysis is analysis

        received_analysis["resolved"][
            "brand_name"
        ] = brand

        received_analysis["resolved"][
            "class_type"
        ] = class_type

        received_analysis[
            "unresolved_required_roles"
        ].clear()

        received_analysis[
            "unresolved"
        ].clear()

        received_analysis[
            "needs_semantics"
        ] = False

        received_analysis[
            "analysis_complete"
        ] = True

        received_analysis[
            "semantic_fields"
        ] = [
            "brand_name",
            "class_type",
        ]

        extraction = {
            "entities": {
                "brand_name": [
                    "JAMESON",
                ],
                "class_type": [
                    "IRISH WHISKEY",
                ],
            },
        }

        return (
            {
                "brand_name": brand,
                "class_type": class_type,
            },
            extraction,
        )

    monkeypatch.setattr(
        analyze_image_module,
        "resolve_semantic_roles",
        fake_resolve_semantic_roles,
    )

    result = (
        analyze_image_module
        .analyze_image(
            image,
            semantic_model="model",
            semantic_schema="schema",
            brand_required=True,
        )
    )

    assert result["status"] == "RESOLVED"
    assert result["reasons"] == []

    assert result[
        "semantic_result"
    ] == {
        "brand_name": brand,
        "class_type": class_type,
    }

    assert result["analysis"][
        "resolved"
    ]["brand_name"] == brand

    assert result["analysis"][
        "resolved"
    ]["class_type"] == class_type

    assert result["analysis"][
        "unresolved_required_roles"
    ] == []

    assert set(
        result["analysis"][
            "semantic_fields"
        ]
    ) == {
        "brand_name",
        "class_type",
    }

