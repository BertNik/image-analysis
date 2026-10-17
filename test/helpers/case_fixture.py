

def field(
    value,
    *,
    unit=None,
    confidence=0.95,
    method="test",
    observed_text=None,
    line_indexes=None,
):
    result = {
        "value": value,
        "confidence": confidence,
        "method": method,
        "line_indexes":
            line_indexes or [0],
    }

    if unit is not None:
        result["unit"] = unit

    if observed_text is not None:
        result[
            "observed_text"
        ] = observed_text

    return result

def usable_image_result(
    *,
    image_index=0,
    filename="label.jpg",
    brand=None,
    class_type=None,
    alcohol=None,
    proof=None,
    net_contents=None,
    businesses=None,
    semantic_fields=None,
    total_seconds=0.1,
):
    return {
        "image_index":
            image_index,

        "filename":
            filename,

        "status":
            "RESOLVED",

        "reasons":
            [],

        "usability": {
            "usable": True,
            "usable_count": 1,
        },

        "timing": {
            "total_seconds":
                total_seconds,
        },

        "analysis": {
            "resolved": {
                "brand_name":
                    brand,

                "class_type":
                    class_type,

                "alcohol_content":
                    alcohol,

                "proof":
                    proof,

                "net_contents":
                    net_contents,

                "businesses":
                    businesses or [],
            },

            "semantic_fields":
                semantic_fields or [],
        },

        "error": None,
    }

def complete_image(
    *,
    image_index=0,
    filename="label.jpg",
):
    return usable_image_result(
        image_index=image_index,
        filename=filename,

        brand=field(
            "OLD TOM DISTILLERY"
        ),

        class_type=field(
            "KENTUCKY STRAIGHT "
            "BOURBON WHISKEY"
        ),

        alcohol=field(
            45,
            unit="percent_abv",
        ),

        proof=field(
            90,
            unit="proof",
        ),

        net_contents=field(
            750,
            unit="mL",
        ),
    )
