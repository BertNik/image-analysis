from __future__ import annotations

from pathlib import Path
from time import perf_counter

import cv2


from .discovery import find_images
from .output import (
    append_jsonl_record,
    initialize_jsonl,
    write_result,
    write_summary,
)
from ..analyze_image import (
    analyze_image as analyze_image_array,
)

def analyze_path(
    image_path: Path,
    *,
    semantic_model=None,
    semantic_schema=None,
    brand_required=None,
) -> dict:
    """
    Load one image from disk and delegate the actual
    label analysis to the canonical image analyzer.
    """

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        return {
            "image": str(image_path),
            "status": "ERROR",
            "reasons": [
                "image_load_failed",
            ],
            "error": "image_load_failed",
        }

    result = analyze_image_array(
        image,
        semantic_model=semantic_model,
        semantic_schema=semantic_schema,
        brand_required=brand_required,
    )

    return {
        "image": str(image_path),
        **result,
    }


def run_batch(
    input_directory: Path,
    output_directory: Path,
    *,
    semantic_model=None,
    semantic_schema=None,
    brand_required=None,
) -> dict:
    """
    Analyze every supported image in a directory.

    The runner owns:
    - discovery
    - iteration
    - error isolation
    - output coordination
    - batch statistics

    The runner does not own label-analysis logic.
    """

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    images = find_images(
        input_directory
    )

    print(
        f"Found {len(images)} images."
    )

    results_path = initialize_jsonl(
        output_directory
    )

    counters = {
        "RESOLVED": 0,
        "VERIFY": 0,
        "SEMANTIC_FALLBACK": 0,
        "ABSTAIN": 0,
        "ERROR": 0,
    }

    total_start = perf_counter()

    for index, image_path in enumerate(
        images,
        start=1,
    ):
        print(
            f"[{index}/{len(images)}] "
            f"{image_path.name}"
        )

        try:
            result = analyze_path(
                image_path,
                semantic_model=semantic_model,
                semantic_schema=semantic_schema,
                brand_required=brand_required,
            )

        except Exception as exc:
            result = {
                "image": str(image_path),
                "status": "ERROR",
                "reasons": [
                    "analysis_error",
                ],
                "error": (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            }

        status = result["status"]

        counters[status] = (
            counters.get(
                status,
                0,
            )
            + 1
        )

        write_result(
            output_directory,
            image_path,
            result,
        )

        append_jsonl_record(
            results_path,
            result,
        )

        print(
            f"    -> {status}"
        )

    elapsed = (
        perf_counter()
        - total_start
    )

    summary = {
        "images": len(images),

        "status_counts": counters,

        "total_seconds": elapsed,

        "mean_seconds_per_image": (
            elapsed / len(images)
            if images
            else 0.0
        ),
    }

    write_summary(
        output_directory,
        summary,
    )

    print()
    print("Batch complete")
    print(summary)

    return summary