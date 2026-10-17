from lib.case.reconcile import reconcile_case

def test_reconcile_case_routes_failed_government_warning_to_verify():
    image_result = {
        "image_index": 0,
        "filename": "label.png",
        "status": "RESOLVED",
        "reasons": [],
        "usability": {
            "usable": True,
        },
        "timing": {
            "total_seconds": 1.0,
        },
        "analysis": {
            "resolved": {
                "brand_name": {
                    "value":
                        "OLD TOM DISTILLERY",
                },
                "class_type": {
                    "value": (
                        "KENTUCKY STRAIGHT "
                        "BOURBON WHISKEY"
                    ),
                },
                "alcohol_content": {
                    "value": 45,
                    "unit":
                        "percent_abv",
                },
                "net_contents": {
                    "value": 750,
                    "unit": "mL",
                },
                "proof": {
                    "value": 90,
                    "unit": "proof",
                },
            },
            "government_warning": {
                "status": "FAIL",
                "reason": (
                    "government_warning_"
                    "heading_not_bold"
                ),
            },
            "semantic_fields": [],
        },
    }

    result = reconcile_case([
        image_result,
    ])

    assert result["status"] == (
        "VERIFY"
    )

    assert result["reasons"] == [
        (
            "government_warning_"
            "heading_not_bold"
        ),
    ]

    assert (
        result["analysis"][
            "analysis_complete"
        ]
        is False
    )