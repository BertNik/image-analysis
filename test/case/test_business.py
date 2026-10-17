from test.helpers.case_fixture import usable_image_result
from lib.case.business import reconcile_businesses

def test_businesses_are_deduplicated_across_images():
    business_a = {
        "name":
            "RIVER OAK DISTILLING CO.",

        "roles": [
            "distiller"
        ],
    }

    business_b = {
        "name":
            "River Oak Distilling Co.",

        "roles": [
            "bottler"
        ],
    }

    result = (
        reconcile_businesses(
            [
                usable_image_result(
                    image_index=0,
                    businesses=[
                        business_a
                    ],
                ),

                usable_image_result(
                    image_index=1,
                    businesses=[
                        business_b
                    ],
                ),
            ]
        )
    )

    assert len(result) == 1

    assert set(
        result[0]["roles"]
    ) == {
        "distiller",
        "bottler",
    }

    assert (
        len(
            result[0]["evidence"]
        )
        == 2
    )


def test_business_without_name_is_ignored():
    result = (
        reconcile_businesses(
            [
                usable_image_result(
                    businesses=[
                        {
                            "name": "",
                            "roles": [
                                "producer"
                            ],
                        }
                    ]
                )
            ]
        )
    )

    assert result == []

