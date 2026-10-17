import json

import numpy as np
import pytest

import lib.batch.runner as runner

@pytest.fixture
def fake_image():
    return np.zeros(
        (20, 20, 3),
        dtype=np.uint8,
    )


@pytest.fixture
def resolved_analysis():
    return {
        "ocr_text": "OLD TOM DISTILLERY",

        "observations": [
            {
                "index": 0,
                "text": "OLD TOM DISTILLERY",
                "confidence": 0.99,
                "box": [
                    0,
                    0,
                    100,
                    20,
                ],
            }
        ],

        "resolved": {
            "brand_name": {
                "value": "OLD TOM DISTILLERY",
            },

            "class_type": {
                "value": "BOURBON WHISKEY",
            },

            "alcohol_content": {
                "value": 45,
                "unit": "percent_abv",
            },

            "proof": {
                "value": 90,
                "unit": "proof",
            },

            "net_contents": {
                "value": 750,
                "unit": "mL",
            },

            "businesses": [],
        },

        "identity_candidates": [],
        "eligible_brand_candidates": [],
        "unresolved_required_fields": [],
        "unresolved_required_roles": [],
        "unresolved": {},
        "needs_semantics": False,
        "deterministic_complete": True,
        "semantic_fields": [],
    }


def resolved_image_result(
    analysis=None,
):
    return {
        "status": "RESOLVED",
        "reasons": [],

        "usability": {
            "usable": True,
            "usable_count": 1,
            "usable_observations": [],
        },

        "timing": {
            "ocr_seconds": 0.1,
            "deterministic_seconds": 0.01,
            "semantic_seconds": 0.0,
            "total_seconds": 0.11,
        },

        "semantic_result": None,

        "analysis": (
            analysis
            if analysis is not None
            else {
                "resolved": {},
            }
        ),

        "error": None,
    }


# ============================================================
# analyze_path
# ============================================================


def test_analyze_path_returns_error_when_image_cannot_load(
    tmp_path,
    monkeypatch,
):
    image_path = tmp_path / "bad.jpg"

    monkeypatch.setattr(
        runner.cv2,
        "imread",
        lambda *args, **kwargs: None,
    )

    result = runner.analyze_path(
        image_path,
        semantic_model=None,
        semantic_schema=None,
    )

    assert result["image"] == str(
        image_path
    )

    assert result["status"] == "ERROR"

    assert result["reasons"] == [
        "image_load_failed"
    ]

    assert (
        result["error"]
        == "image_load_failed"
    )


def test_analyze_path_resolved_without_semantics(
    tmp_path,
    monkeypatch,
    fake_image,
    resolved_analysis,
):
    image_path = tmp_path / "label.jpg"
    image_path.write_bytes(b"fake")
    


    monkeypatch.setattr(
        runner.cv2,
        "imread",
        lambda *args, **kwargs:
            fake_image,
    )

    captured = {}
    called = False
    
    def fake_analyze_image(
        image,
        *,
        semantic_model=None,
        semantic_schema=None,
        brand_required=False,
    ):
        nonlocal called
        called = True

        captured["image"] = image
        captured["semantic_model"] = semantic_model
        captured["semantic_schema"] = semantic_schema
        captured["brand_required"] = brand_required

        return resolved_image_result(
            resolved_analysis
        )

    # # batch_runner imports lib.analyze_image as a module.
    # # Therefore, patch the function inside that module.
    monkeypatch.setattr(
        runner,
        "analyze_image_array",
        fake_analyze_image,
    )

    # Invoke the real filesystem adapter.
    result = runner.analyze_path(
        image_path,
        semantic_model="model",
        semantic_schema="schema",
        brand_required=True,
    )

    assert "status" in result, (
        "analyze_path() returned an invalid "
        f"result: {result!r}"
    )

    assert result["status"] == "RESOLVED"
    assert result["image"] == str(
        image_path
    )

    assert result["analysis"] is (
        resolved_analysis
    )

    assert (
        result["semantic_result"]
        is None
    )

    assert (
        captured["image"]
        is fake_image
    )

    assert (
        captured["semantic_model"]
        == "model"
    )

    assert (
        captured["semantic_schema"]
        == "schema"
    )

    assert (
        captured["brand_required"]
        is True
    )


def test_analyze_path_passes_brand_required_false(
    tmp_path,
    monkeypatch,
    fake_image,
):
    image_path = tmp_path / "label.jpg"
    image_path.write_bytes(b"fake")

    monkeypatch.setattr(
        runner.cv2,
        "imread",
        lambda *args, **kwargs: fake_image,
    )

    captured = {}

    def fake_analyze_image(
        image,
        *,
        semantic_model=None,
        semantic_schema=None,
        brand_required=False,
    ):
        captured["brand_required"] = brand_required

        return {
            "status": "RESOLVED",
            "reasons": [],
            "usability": {
                "usable": True,
            },
            "timing": {
                "ocr_seconds": 0.0,
                "deterministic_seconds": 0.0,
                "semantic_seconds": 0.0,
                "total_seconds": 0.0,
            },
            "semantic_result": None,
            "analysis": {
                "resolved": {},
            },
            "error": None,
        }

    monkeypatch.setattr(
        runner,
        "analyze_image_array",
        fake_analyze_image,
    )

    result = runner.analyze_path(
        image_path,
        semantic_model=None,
        semantic_schema=None,
        brand_required=False,
    )

    assert result["status"] == "RESOLVED"
    assert result["image"] == str(image_path)
    assert captured["brand_required"] is False

def test_analyze_path_passes_brand_required_true(
    tmp_path,
    monkeypatch,
    fake_image,
):
    image_path = tmp_path / "label.jpg"
    image_path.write_bytes(b"fake")

    monkeypatch.setattr(
        runner.cv2,
        "imread",
        lambda *args, **kwargs:
            fake_image,
    )

    captured = {}

    def fake_analyze_image(
        image,
        *,
        semantic_model=None,
        semantic_schema=None,
        brand_required=False,
    ):
        captured["brand_required"] = (
            brand_required
        )

        return {
            "status": "RESOLVED",
            "reasons": [],
            "usability": {
                "usable": True,
            },
            "timing": {},
            "semantic_result": None,
            "analysis": {},
            "error": None,
        }

    monkeypatch.setattr(
        runner,
        "analyze_image_array",
        fake_analyze_image,
    )

    result = runner.analyze_path(
        image_path,
        semantic_model=None,
        semantic_schema=None,
        brand_required=True,
    )

    assert (
        captured["brand_required"]
        is True
    )

    assert result["status"] == "RESOLVED"
    assert result["image"] == str(
        image_path
    )

def test_analyze_path_preserves_canonical_result(
    tmp_path,
    monkeypatch,
    fake_image,
):
    image_path = tmp_path / "label.jpg"
    image_path.write_bytes(b"fake")

    monkeypatch.setattr(
        runner.cv2,
        "imread",
        lambda *args, **kwargs: fake_image,
    )

    canonical_result = {
        "status": "VERIFY",
        "reasons": [
            "semantic_brand_unresolved",
        ],
        "usability": {
            "usable": True,
        },
        "timing": {
            "semantic_seconds": 0.02,
            "total_seconds": 0.12,
        },
        "semantic_result": None,
        "analysis": {
            "resolved": {
                "brand_name": None,
            },
        },
        "error": None,
    }

    monkeypatch.setattr(
        runner,
        "analyze_image_array",
        lambda *args, **kwargs:
            canonical_result,
    )

    result = runner.analyze_path(
        image_path,
        semantic_model="model",
        semantic_schema="schema",
    )

    assert result["image"] == str(image_path)
    assert result["status"] == "VERIFY"

    assert result["reasons"] == [
        "semantic_brand_unresolved",
    ]

    assert (
        result["analysis"]
        == canonical_result["analysis"]
    )


# ============================================================
# run_batch
# ============================================================


def test_run_batch_writes_results_jsonl_and_summary(
    tmp_path,
    monkeypatch,
    capsys,
):
    input_directory = (
        tmp_path / "input"
    )

    output_directory = (
        tmp_path / "output"
    )

    input_directory.mkdir()

    images = [
        input_directory / "a.jpg",
        input_directory / "b.png",
    ]

    for path in images:
        path.write_bytes(b"fake")

    calls = []

    def fake_analyze_path(
        image_path,
        *,
        semantic_model=None,
        semantic_schema=None,
        brand_required=False,
    ):
        calls.append(
            {
                "image_path":
                    image_path,

                "semantic_model":
                    semantic_model,

                "semantic_schema":
                    semantic_schema,

                "brand_required":
                    brand_required,
            }
        )

        status = (
            "RESOLVED"
            if image_path.name == "a.jpg"
            else "VERIFY"
        )

        return {
            "image": str(image_path),

            "status": status,

            "reasons": (
                []
                if status == "RESOLVED"
                else [
                    "brand_name_not_resolved"
                ]
            ),

            "usability": {
                "usable": True,
            },

            "timing": {
                "total_seconds": 0.1,
            },

            "semantic_result": None,

            "analysis": {
                "resolved": {
                    "brand_name": (
                        {
                            "value":
                                image_path.stem,
                        }
                        if status == "RESOLVED"
                        else None
                    ),
                },

                "identity_candidates": [],

                "unresolved_required_fields": (
                    []
                    if status == "RESOLVED"
                    else [
                        "brand_name"
                    ]
                ),

                "unresolved_required_roles": [],
            },

            "error": None,
        }

    # run_batch looks up analyze_path on batch_runner.
    monkeypatch.setattr(
        runner,
        "analyze_path",
        fake_analyze_path,
    )

    summary = runner.run_batch(
        input_directory=
            input_directory,

        output_directory=
            output_directory,

        semantic_model="model",
        semantic_schema="schema",
        brand_required=True,
    )

    assert len(calls) == 2

    assert [
        call["image_path"]
        for call in calls
    ] == images

    assert all(
        call["semantic_model"]
        == "model"
        for call in calls
    )

    assert all(
        call["semantic_schema"]
        == "schema"
        for call in calls
    )

    assert all(
        call["brand_required"]
        is True
        for call in calls
    )

    assert (
        output_directory
        / "a.json"
    ).exists()

    assert (
        output_directory
        / "b.json"
    ).exists()

    results_path = (
        output_directory
        / "results.jsonl"
    )

    assert results_path.exists()

    records = [
        json.loads(line)
        for line in (
            results_path
            .read_text(
                encoding="utf-8"
            )
            .splitlines()
        )
    ]

    assert len(records) == 2

    assert {
        record["status"]
        for record in records
    } == {
        "RESOLVED",
        "VERIFY",
    }

    assert (
        summary["images"]
        == 2
    )

    assert (
        summary["status_counts"][
            "RESOLVED"
        ]
        == 1
    )

    assert (
        summary["status_counts"][
            "VERIFY"
        ]
        == 1
    )

    assert (
        summary["status_counts"][
            "ERROR"
        ]
        == 0
    )

    assert (
        summary[
            "mean_seconds_per_image"
        ]
        >= 0.0
    )

    written_summary = json.loads(
        (
            output_directory
            / "summary.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert (
        written_summary[
            "images"
        ]
        == 2
    )

    output = capsys.readouterr().out

    assert "Found 2 images." in output
    assert "Batch complete" in output


def test_run_batch_handles_empty_directory(
    tmp_path,
):
    input_directory = (
        tmp_path / "empty"
    )

    output_directory = (
        tmp_path / "output"
    )

    input_directory.mkdir()

    summary = runner.run_batch(
        input_directory=
            input_directory,

        output_directory=
            output_directory,

        semantic_model=None,
        semantic_schema=None,
    )

    assert summary["images"] == 0

    assert (
        summary[
            "mean_seconds_per_image"
        ]
        == 0.0
    )

    assert (
        summary["status_counts"][
            "RESOLVED"
        ]
        == 0
    )

    assert (
        summary["status_counts"][
            "VERIFY"
        ]
        == 0
    )

    assert (
        summary["status_counts"][
            "ERROR"
        ]
        == 0
    )

    assert (
        output_directory
        / "results.jsonl"
    ).read_text(
        encoding="utf-8"
    ) == ""


def test_run_batch_continues_after_single_image_failure(
    tmp_path,
    monkeypatch,
):
    input_directory = (
        tmp_path / "input"
    )

    output_directory = (
        tmp_path / "output"
    )

    input_directory.mkdir()

    bad = input_directory / "a.jpg"
    good = input_directory / "b.jpg"

    bad.write_bytes(b"bad")
    good.write_bytes(b"good")

    def fake_analyze_path(
        image_path,
        *,
        semantic_model=None,
        semantic_schema=None,
        brand_required=False,
    ):
        if image_path.name == "a.jpg":
            raise RuntimeError(
                "simulated failure"
            )

        return {
            "image": str(image_path),

            "status": "RESOLVED",

            "reasons": [],

            "usability": {
                "usable": True,
            },

            "timing": {
                "total_seconds": 0.1,
            },

            "semantic_result": None,

            "analysis": {
                "resolved": {},
                "identity_candidates": [],
                "unresolved_required_fields": [],
                "unresolved_required_roles": [],
            },

            "error": None,
        }

    monkeypatch.setattr(
        runner,
        "analyze_path",
        fake_analyze_path,
    )

    summary = runner.run_batch(
        input_directory=
            input_directory,

        output_directory=
            output_directory,

        semantic_model=None,
        semantic_schema=None,
    )

    assert (
        summary["status_counts"][
            "ERROR"
        ]
        == 1
    )

    assert (
        summary["status_counts"][
            "RESOLVED"
        ]
        == 1
    )

    records = [
        json.loads(line)
        for line in (
            output_directory
            / "results.jsonl"
        ).read_text(
            encoding="utf-8"
        ).splitlines()
    ]

    assert len(records) == 2

    error_records = [
        record
        for record in records
        if record["status"]
        == "ERROR"
    ]

    assert len(error_records) == 1

    assert (
        "RuntimeError: simulated failure"
        in error_records[0]["error"]
    )


def test_run_batch_counts_abstain_status(
    tmp_path,
    monkeypatch,
):
    input_directory = (
        tmp_path / "input"
    )

    output_directory = (
        tmp_path / "output"
    )

    input_directory.mkdir()

    image_path = (
        input_directory / "a.jpg"
    )

    image_path.write_bytes(b"fake")

    monkeypatch.setattr(
        runner,
        "analyze_path",
        lambda image_path,
        **kwargs: {
            "image": str(image_path),

            "status": "ABSTAIN",

            "reasons": [
                "no_usable_ocr_evidence"
            ],

            "usability": {
                "usable": False,
            },

            "timing": {
                "total_seconds": 0.1,
            },

            "semantic_result": None,
            "analysis": {},
            "error": None,
        },
    )

    summary = runner.run_batch(
        input_directory=
            input_directory,

        output_directory=
            output_directory,

        semantic_model=None,
        semantic_schema=None,
    )

    assert (
        summary["status_counts"][
            "ABSTAIN"
        ]
        == 1
    )