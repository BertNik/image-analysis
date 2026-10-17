from __future__ import annotations

from typing import Any

def _entity_text(item: Any) -> str:
    if isinstance(item, str):
        value = item
    elif isinstance(item, dict):
        value = (
            item.get("text")
            or item.get("value")
            or item.get("span")
            or ""
        )
    else:
        value = str(item or "")

    return str(value or "").strip()


def get_entity_values(
    extraction: dict[str, Any],
    label: str,
) -> list[str]:
    """Normalize supported GLiNER entity-output shapes into unique strings."""
    if not isinstance(extraction, dict):
        return []

    entities = extraction.get("entities", {})
    if not isinstance(entities, dict):
        return []

    raw_values = entities.get(label, [])
    if raw_values is None:
        return []

    # A single dictionary is one structured entity, not an iterable of keys.
    if isinstance(raw_values, (str, dict)):
        raw_values = [raw_values]
    elif not isinstance(raw_values, (list, tuple)):
        raw_values = [raw_values]

    values: list[str] = []
    seen: set[str] = set()

    for item in raw_values:
        value = _entity_text(item)
        if not value or value in seen:
            continue
        seen.add(value)
        values.append(value)

    return values


def normalize_relation_pair(value: Any) -> tuple[str, str] | None:
    """Normalize common GLiNER relation-pair representations."""
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            return None
        source, target = value
    elif isinstance(value, dict):
        source = (
            value.get("source")
            or value.get("head")
            or value.get("subject")
            or value.get("from")
        )
        target = (
            value.get("target")
            or value.get("tail")
            or value.get("object")
            or value.get("to")
        )
    else:
        return None

    source = str(source or "").strip()
    target = str(target or "").strip()
    if not source or not target:
        return None

    return source, target


def _as_relation_items(raw_pairs: Any) -> list[Any]:
    if raw_pairs is None:
        return []

    if isinstance(raw_pairs, dict):
        return [raw_pairs]

    if isinstance(raw_pairs, (list, tuple)):
        # A direct scalar pair should be handled as one relation.
        if (
            len(raw_pairs) == 2
            and not any(
                isinstance(item, (list, tuple, dict))
                for item in raw_pairs
            )
        ):
            return [raw_pairs]
        return list(raw_pairs)

    return [raw_pairs]


def get_relation_pairs(
    extraction: dict[str, Any],
    relation_name: str,
) -> list[tuple[str, str]]:
    """Return unique normalized pairs for one canonical relation type."""
    if not isinstance(extraction, dict):
        return []

    relations = extraction.get("relation_extraction", {})
    if not isinstance(relations, dict):
        return []

    requested_name = canonical_relation_name(relation_name)
    if not requested_name:
        return []

    results: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for returned_name, raw_pairs in relations.items():
        if canonical_relation_name(returned_name) != requested_name:
            continue

        for raw_pair in _as_relation_items(raw_pairs):
            pair = normalize_relation_pair(raw_pair)
            if pair is None or pair in seen:
                continue
            seen.add(pair)
            results.append(pair)

    return results


def ensure_extraction(
    model,
    schema,
    analysis: dict[str, Any],
    extraction: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Use a supplied extraction or perform one extraction for usable OCR."""
    if extraction is not None:
        return extraction

    observations = analysis.get("observations", [])
    ocr_text = str(analysis.get("ocr_text", "") or "").strip()
    if not observations or not ocr_text:
        return None

    return model.extract(ocr_text, schema)


def canonical_relation_name(value: Any) -> str:
    return str(value or "").split(":", 1)[0].strip()

