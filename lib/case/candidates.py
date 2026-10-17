

def collect_case_candidates(
    image_results,
    field_name,
):
    candidate_key = (
        f"semantic_{field_name}_candidates"
    )

    candidates = []
    seen = set()

    for result in image_results:
        image_index = result.get(
            "image_index"
        )
        filename = result.get(
            "filename"
        )

        image_analysis = result.get(
            "analysis",
            {},
        )

        for candidate in (
            image_analysis.get(
                candidate_key,
                [],
            )
        ):
            value = candidate.get(
                "value"
            )

            if not value:
                continue

            key = (
                str(value),
                image_index,
                tuple(
                    candidate.get(
                        "line_indexes",
                        [],
                    )
                ),
            )

            if key in seen:
                continue

            seen.add(key)

            candidates.append(
                {
                    **candidate,
                    "image_index":
                        image_index,
                    "filename":
                        filename,
                }
            )

    return candidates