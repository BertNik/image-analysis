

class FakeModel:
    def __init__(
        self,
        extraction=None,
    ):
        self.extraction = (
            extraction
            if extraction is not None
            else {
                "entities": {},
            }
        )

        self.extract_calls = []

    def extract(
        self,
        text,
        schema,
    ):
        self.extract_calls.append(
            {
                "text": text,
                "schema": schema,
            }
        )

        return self.extraction

class FakeSchemaBuilder:
    def __init__(self):
        self.entity_definition = None
        self.relation_definition = None

    def entities(
        self,
        definition,
    ):
        self.entity_definition = (
            definition
        )

        return self

    def relations(
        self,
        definition,
    ):
        self.relation_definition = (
            definition
        )

        return self

class FakeSchemaModel:
    def __init__(self):
        self.builder = (
            FakeSchemaBuilder()
        )

    def create_schema(self):
        return self.builder


def candidate(
    value,
    confidence=0.95,
    line_indexes=None,
):
    return {
        "value": value,

        "confidence":
            confidence,

        "line_indexes":
            (
                line_indexes
                if line_indexes
                is not None
                else [0]
            ),
    }


def semantic_analysis(
    *,
    candidates=None,
    ocr_text="OLD TOM DISTILLERY",
    unresolved_roles=None,
    class_type=None,
):
    candidates = (
        candidates
        if candidates is not None
        else []
    )

    observations = [
        {
            "index": (
                candidate.get(
                    "line_indexes",
                    [index],
                )[0]
            ),
            "text":
                candidate.get(
                    "value",
                    "",
                ),
            "confidence":
                candidate.get(
                    "confidence"
                ),
            "box":
                None,
        }
        for index, candidate
        in enumerate(candidates)
    ]

    return {
        "ocr_text":
            ocr_text,

        "observations":
            observations,

        "eligible_brand_candidates":
            candidates,

        "unresolved_required_roles": (
            unresolved_roles
            if unresolved_roles is not None
            else ["brand_name"]
        ),

        "unresolved_required_fields":
            [],

        "resolved": {
            "brand_name":
                None,

            "class_type": (
                {
                    "value":
                        class_type,
                }
                if class_type
                else None
            ),
        },

        "unresolved": {
            "brand_name": {
                "reason":
                    "brand_requires_semantic_resolution",

                "candidates":
                    candidates,
            }
        },

        "needs_semantics":
            bool(
                unresolved_roles
                if unresolved_roles is not None
                else ["brand_name"]
            ),

        "analysis_complete":
            False,

        "deterministic_complete":
            False,

        "semantic_fields":
            [],
    }