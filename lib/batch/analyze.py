from __future__ import annotations

import argparse
from pathlib import Path

from .runner import run_batch
from ..resolver.semantic import (
    create_brand_schema,
    load_semantic_model,
)


def parse_args():
    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "input_directory",
        type=Path,
        help=(
            "Directory containing "
            "label images"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "analysis_output"
        ),
    )

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--brand-required",
        action="store_true",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    semantic_model = (
        load_semantic_model(
            args.model
        )
    )

    semantic_schema = (
        create_brand_schema(
            semantic_model
        )
    )

    run_batch(
        input_directory=
            args.input_directory,

        output_directory=
            args.output,

        semantic_model=
            semantic_model,

        semantic_schema=
            semantic_schema,

        brand_required=
            args.brand_required,
    )


if __name__ == "__main__":
    main()