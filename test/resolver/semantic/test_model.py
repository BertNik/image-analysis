import pytest

import lib.resolver.semantic.model as model_module

from lib.resolver.semantic.model import (
    create_brand_schema,
    create_label_schema,
    load_semantic_model,
)
from lib.resolver.semantic.schema import (
    BUSINESS_RELATIONS,
    LABEL_ENTITIES,
)

from test.resolver.helpers import helpers


def test_load_semantic_model_rejects_missing_directory(
    tmp_path,
):
    missing = (
        tmp_path / "missing-model"
    )

    with pytest.raises(
        FileNotFoundError,
        match=(
            "Model directory "
            "does not exist"
        ),
    ):
        (
            load_semantic_model(
                missing
            )
        )


def test_load_semantic_model_uses_resolved_path(
    tmp_path,
    monkeypatch,
    capsys,
):
    model_directory = (
        tmp_path / "model"
    )

    model_directory.mkdir()

    loaded_model = object()
    received_calls = []

    def fake_from_pretrained(
        path,
        *,
        map_location=None,
    ):
        received_calls.append(
            {
                "path": path,
                "map_location":
                    map_location,
            }
        )

        return loaded_model

    monkeypatch.setattr(
        model_module.AutoExtractor,
        "from_pretrained",
        fake_from_pretrained,
    )

    result = (
        load_semantic_model(
            model_directory
        )
    )

    assert result is loaded_model

    assert received_calls == [
        {
            "path": str(
                model_directory
                .resolve()
            ),
            "map_location":
                "cuda",
        }
    ]

    output = (
        capsys
        .readouterr()
        .out
    )

    assert (
        str(
            model_directory.resolve()
        )
        in output
    )



def test_create_label_schema_uses_entities_and_relations():
    model = helpers.FakeSchemaModel()

    result = (
        create_label_schema(
            model
        )
    )

    assert result is model.builder

    assert (
        model.builder.entity_definition
        ==
        LABEL_ENTITIES
    )

    assert (
        model.builder.relation_definition
        ==
        BUSINESS_RELATIONS
    )

def test_create_brand_schema_is_compatibility_alias():
    model = helpers.FakeSchemaModel()

    result = (
        create_brand_schema(
            model
        )
    )

    assert result is model.builder

    assert (
        model.builder.entity_definition
        ==
        LABEL_ENTITIES
    )

    assert (
        model.builder.relation_definition
        ==
        BUSINESS_RELATIONS
    )
