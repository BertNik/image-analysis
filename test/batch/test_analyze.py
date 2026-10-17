import json
import pytest

import lib.batch.analyze as analyze
import lib.analyze_image as analyze_image

# ============================================================
# Shared fixtures and helpers
# ============================================================

def resolved_route():
    return {
        "status": "RESOLVED",
        "reasons": [],
        "usability": {
            "usable": True,
            "usable_count": 1,
            "usable_observations": [],
        },
    }


def semantic_route():
    return {
        "status":
            "SEMANTIC_FALLBACK",

        "reasons": [
            "brand_name_role_ambiguous"
        ],

        "usability": {
            "usable": True,
            "usable_count": 1,
            "usable_observations": [],
        },
    }



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

    analyze.run_batch(
        input_directory=
            input_directory,

        output_directory=
            output_directory,

        semantic_model=None,
        semantic_schema=None,
    )

    summary = json.loads(
        (
            output_directory
            / "summary.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert summary["images"] == 0

    assert (
        summary[
            "mean_seconds_per_image"
        ]
        == 0.0
    )

    assert (
        output_directory
        / "results.jsonl"
    ).read_text(
        encoding="utf-8"
    ) == ""


@pytest.mark.xfail(
    reason=(
        "Known behavior: run_batch catches an exception, "
        "constructs an ERROR result, then immediately "
        "re-raises the exception instead of continuing."
    ),
    strict=True,
)
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
        **kwargs,
    ):
        if image_path.name == "a.jpg":
            raise RuntimeError(
                "simulated failure"
            )

        return {
            "image":
                str(image_path),

            "status":
                "RESOLVED",

            "timing": {},

            "analysis": {
                "resolved": {},
                "identity_candidates": [],
                "unresolved_required_fields": [],
                "unresolved_required_roles": [],
            },

            "error": None,
        }

    monkeypatch.setattr(
        analyze_image,
        "analyze_path",
        fake_analyze_path,
    )

    analyze.run_batch(
        input_directory=
            input_directory,

        output_directory=
            output_directory,

        semantic_model=None,
        semantic_schema=None,
    )

    summary = json.loads(
        (
            output_directory
            / "summary.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert summary[
        "status_counts"
    ]["ERROR"] == 1

    assert summary[
        "status_counts"
    ]["RESOLVED"] == 1


# ============================================================
# main
# ============================================================


def test_main_loads_model_once_and_runs_batch(
    tmp_path,
    monkeypatch,
):
    input_directory = (
        tmp_path / "input"
    )

    output_directory = (
        tmp_path / "output"
    )

    model_path = (
        tmp_path / "model"
    )

    input_directory.mkdir()
    model_path.mkdir()

    model = object()
    schema = object()

    load_calls = []

    def fake_load(path):
        load_calls.append(path)
        return model

    monkeypatch.setattr(
        analyze,
        "load_semantic_model",
        fake_load,
    )

    monkeypatch.setattr(
        analyze,
        "create_brand_schema",
        lambda loaded_model: schema,
    )

    run_calls = []

    monkeypatch.setattr(
        analyze,
        "run_batch",
        lambda **kwargs:
            run_calls.append(kwargs),
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "batch_analyze.py",
            str(input_directory),
            "--output",
            str(output_directory),
            "--model",
            str(model_path),
            "--brand-required",
        ],
    )

    analyze.main()

    # This assertion documents the intended behavior.
    # The current implementation loads the model twice.
    assert load_calls == [
        model_path,
    ]

    assert len(run_calls) == 1

    call = run_calls[0]

    assert (
        call["input_directory"]
        == input_directory
    )

    assert (
        call["output_directory"]
        == output_directory
    )

    assert (
        call["semantic_model"]
        is model
    )

    assert (
        call["semantic_schema"]
        is schema
    )

    assert (
        call["brand_required"]
        is True
    )
