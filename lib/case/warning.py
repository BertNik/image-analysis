from typing import Any

def government_warning_reasons(
    image_results: list[
        dict[str, Any]
    ],
) -> list[str]:
    reasons = []

    for result in image_results:
        warning = (
            result
            .get("analysis", {})
            .get(
                "government_warning",
                {},
            )
        )

        if not isinstance(
            warning,
            dict,
        ):
            continue

        if (
            warning.get("status")
            != "FAIL"
        ):
            continue

        reason = warning.get(
            "reason"
        )

        if (
            isinstance(reason, str)
            and reason
            and reason not in reasons
        ):
            reasons.append(
                reason
            )

    return reasons