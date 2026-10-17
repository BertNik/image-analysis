import json
from pathlib import Path
from typing import Any


def build_compact_record(
    result: dict[str, Any],
) -> dict[str, Any]:
    analysis = result.get(
        "analysis",
        {},
    )

    return {
        "image":
            result.get("image"),

        "status":
            result.get("status"),

        "timing":
            result.get("timing"),

        "resolved":
            analysis.get("resolved"),

        "identity_candidates":
            analysis.get(
                "identity_candidates"
            ),

        "unresolved_required_fields":
            analysis.get(
                "unresolved_required_fields"
            ),

        "unresolved_required_roles":
            analysis.get(
                "unresolved_required_roles"
            ),

        "error":
            result.get("error"),
    }


def write_result(
    output_directory: Path,
    image_path: Path,
    result: dict[str, Any],
) -> None:
    result_path = (
        output_directory
        / f"{image_path.stem}.json"
    )

    with result_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=2,
            ensure_ascii=False,
        )


def initialize_jsonl(
    output_directory: Path,
) -> Path:
    path = (
        output_directory
        / "results.jsonl"
    )

    path.write_text(
        "",
        encoding="utf-8",
    )

    return path


def append_jsonl_record(
    path: Path,
    result: dict[str, Any],
) -> None:
    record = build_compact_record(
        result
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


def write_summary(
    output_directory: Path,
    summary: dict[str, Any],
) -> None:
    path = (
        output_directory
        / "summary.json"
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )
        