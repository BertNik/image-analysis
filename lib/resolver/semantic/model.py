from __future__ import annotations

from pathlib import Path

from gliner2 import AutoExtractor

from .schema import (
    BUSINESS_RELATIONS,
    LABEL_ENTITIES,
)


def load_semantic_model(model_path: str | Path):
    model_path = Path(model_path).resolve()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model directory does not exist: {model_path}"
        )

    print(f"Loading semantic model from: {model_path}")
    return AutoExtractor.from_pretrained(
        str(model_path),  map_location="cpu", 
    )


def create_label_schema(model):
    return (
        model.create_schema()
        .entities(LABEL_ENTITIES)
        .relations(BUSINESS_RELATIONS)
    )


def create_brand_schema(model):
    """Compatibility alias for callers using the earlier schema name."""
    return create_label_schema(model)