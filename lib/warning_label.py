from typing import Any

from lib.resolver.visual import (
    belongs_to_warning_column,
    get_box
)

GOVERNMENT_WARNING_HEADING = (
    "GOVERNMENT WARNING:"
)


GOVERNMENT_WARNING_TEXT = (
    "GOVERNMENT WARNING: "
    "(1) According to the Surgeon General, "
    "women should not drink alcoholic beverages "
    "during pregnancy because of the risk of "
    "birth defects. "
    "(2) Consumption of alcoholic beverages "
    "impairs your ability to drive a car or "
    "operate machinery, and may cause health "
    "problems."
)


WARNING_END_TEXT = ( "MAY CAUSE HEALTH PROBLEMS." )

def normalize_warning_text(
    value: str,
    *,
    case_sensitive: bool = True,
) -> str:
    normalized = " ".join(
        str(value or "").split()
    )

    if not case_sensitive:
        normalized = (
            normalized.casefold()
        )

    return normalized



def extract_government_warning(
    observations: list[dict[str, Any]],
) -> dict[str, Any] | None:
    heading_position = None
    heading_observation = None

    for position, observation in enumerate(
        observations
    ):
        text = str(
            observation.get(
                "text",
                "",
            )
        ).strip()

        if text == GOVERNMENT_WARNING_HEADING:
            heading_position = position
            heading_observation = observation
            break

    if (
        heading_position is None
        or heading_observation is None
    ):
        return None

    heading_box = get_box(
        heading_observation
    )

    if heading_box is None:
        return None

    warning_observations = [
        heading_observation
    ]

    complete = False

    for observation in observations[
        heading_position + 1:
    ]:
        text = str(
            observation.get(
                "text",
                "",
            )
        ).strip()

        if not text:
            continue

        if not belongs_to_warning_column(
            observation,
            heading_box,
        ):
            continue

        warning_observations.append(
            observation
        )

        candidate_text = " ".join(
            str(
                item.get(
                    "text",
                    "",
                )
            ).strip()
            for item in warning_observations
        )

        normalized_candidate = (
            " ".join(
                candidate_text
                .upper()
                .split()
            )
        )

        if (
            WARNING_END_TEXT
            in normalized_candidate
        ):
            complete = True
            break
       

    observed_text = "\n".join(
        str(
            observation.get(
                "text",
                "",
            )
        ).strip()
        for observation
        in warning_observations
    )

    confidences = [
        float(
            observation.get(
                "confidence",
                0.0,
            )
            or 0.0
        )
        for observation
        in warning_observations
    ]

    return {
        "value":
            observed_text,

        "observed_text":
            observed_text,

        "heading_text":
            str(
                heading_observation.get(
                    "text",
                    "",
                )
            ).strip(),

        "heading_box":
            list(heading_box),

        "line_indexes": [
            observation["index"]
            for observation
            in warning_observations
            if observation.get("index")
            is not None
        ],

        "confidence": (
            min(confidences)
            if confidences
            else 0.0
        ),

        "complete":
            complete,

        "heading_bold":
            None,

        "evidence": [
            {
                "line_index":
                    observation.get(
                        "index"
                    ),
                "text":
                    observation.get(
                        "text"
                    ),
                "confidence":
                    observation.get(
                        "confidence"
                    ),
                "box":
                    observation.get(
                        "box"
                    ),
            }
            for observation
            in warning_observations
        ],

        "method":
            "spatial_warning_extraction",
    }

def validate_government_warning(
    warning: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Validate an extracted Government Health Warning.

    PASS:
        Exact text, uppercase heading, and bold heading
        have all been established.

    FAIL:
        Warning is absent, incomplete, wording differs,
        heading capitalization is wrong, or boldness is
        explicitly false.

    VERIFY:
        Text requirements pass but heading boldness cannot
        be established from available evidence.
    """
    if warning is None:
        return {
            "status":
                "FAIL",

            "observed":
                False,

            "complete":
                False,

            "text_exact":
                False,

            "heading_uppercase":
                False,

            "heading_bold":
                None,

            "value":
                None,

            "observed_text":
                None,

            "line_indexes":
                [],

            "confidence":
                None,

            "evidence":
                [],

            "reason":
                "government_warning_not_observed",

            "method":
                "deterministic_warning_validation",
        }

    observed_text = normalize_warning_text(
        warning.get(
            "observed_text",
            "",
        ),
        case_sensitive=False,
    )

    expected_text = normalize_warning_text(
        GOVERNMENT_WARNING_TEXT,
        case_sensitive=False,
    )

    complete = bool(
        warning.get(
            "complete",
            False,
        )
    )

    text_exact = (
        complete
        and observed_text == expected_text
    )

    heading_text = str(
        warning.get(
            "heading_text",
            "",
        )
    ).strip()

    heading_uppercase = (
        heading_text
        == GOVERNMENT_WARNING_HEADING
    )

    heading_bold = warning.get(
        "heading_bold"
    )

    if (
        text_exact
        and heading_uppercase
        and heading_bold is True
    ):
        status = "PASS"
        reason = None

    elif not complete:
        status = "FAIL"
        reason = (
            "government_warning_incomplete"
        )

    elif not text_exact:
        status = "FAIL"
        reason = (
            "government_warning_text_mismatch"
        )

    elif not heading_uppercase:
        status = "FAIL"
        reason = (
            "government_warning_heading_not_uppercase"
        )

    elif heading_bold is False:
        status = "FAIL"
        reason = (
            "government_warning_heading_not_bold"
        )

    else:
        status = "VERIFY"
        reason = (
            "government_warning_boldness_unresolved"
        )

    return {
        **warning,

        "status":
            status,

        "observed":
            True,

        "complete":
            complete,

        "text_exact":
            text_exact,

        "heading_uppercase":
            heading_uppercase,

        "heading_bold":
            heading_bold,

        "reason":
            reason,

        "method":
            "visual_warning_validation",
    }
