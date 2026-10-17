from lib.resolver.semantic import (
    create_label_schema,
    load_semantic_model,
)

SEMANTIC_MODEL = None
SEMANTIC_SCHEMA = None


def initialize_semantics(
    model_path,
):
    global SEMANTIC_MODEL
    global SEMANTIC_SCHEMA

    SEMANTIC_MODEL = (
        load_semantic_model(
            model_path
        )
    )

    SEMANTIC_SCHEMA = (
        create_label_schema(
            SEMANTIC_MODEL
        )
    )

    return SEMANTIC_MODEL, SEMANTIC_SCHEMA
