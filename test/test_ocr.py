import numpy as np
import pytest


import lib.ocr as ocr


# ============================================================
# Fake PaddleOCR result
# ============================================================


class FakeResult:
    def __init__(
        self,
        texts=None,
        scores=None,
        boxes=None,
    ):
        self.json = {
            "res": {
                "rec_texts":
                    texts or [],

                "rec_scores":
                    scores or [],

                "rec_boxes":
                    boxes or [],
            }
        }


# ============================================================
# No scaling
# ============================================================


def test_ocr_function_returns_expected_structure(
    monkeypatch,
):
    image = np.zeros(
        (100, 200, 3),
        dtype=np.uint8,
    )

    monkeypatch.setattr(
        ocr.ocr_engine,
        "predict",
        lambda image: [
            FakeResult(
                texts=[
                    "JAMESON",
                    "IRISH WHISKEY",
                ],
                scores=[
                    0.99,
                    0.95,
                ],
                boxes=[
                    [10, 10, 100, 30],
                    [20, 40, 150, 60],
                ],
            )
        ],
    )

    result = ocr.ocr_function(
        image
    )

    assert result["text"] == (
        "JAMESON\n"
        "IRISH WHISKEY"
    )

    assert result["lines"] == [
        "JAMESON",
        "IRISH WHISKEY",
    ]

    assert result["scores"] == [
        0.99,
        0.95,
    ]

    assert result["boxes"] == [
        [10, 10, 100, 30],
        [20, 40, 150, 60],
    ]


# ============================================================
# Original image dimensions
# ============================================================


def test_ocr_function_records_original_image_space(
    monkeypatch,
):
    image = np.zeros(
        (100, 200, 3),
        dtype=np.uint8,
    )

    monkeypatch.setattr(
        ocr.ocr_engine,
        "predict",
        lambda image: [],
    )

    result = ocr.ocr_function(
        image
    )

    image_space = result[
        "image_space"
    ]

    assert image_space["scale"] == 1.0

    assert image_space["original"] == {
        "width": 200,
        "height": 100,
        "pixels": 20000,
    }

    assert image_space["processed"] == {
        "width": 200,
        "height": 100,
        "pixels": 20000,
    }

    assert (
        image_space[
            "pixel_area_scale"
        ]
        == pytest.approx(1.0)
    )


# ============================================================
# Scaling
# ============================================================


def test_ocr_function_scales_image_before_prediction(
    monkeypatch,
):
    image = np.zeros(
        (100, 200, 3),
        dtype=np.uint8,
    )

    received = []

    def fake_predict(
        processed_image,
    ):
        received.append(
            processed_image.shape
        )

        return []

    monkeypatch.setattr(
        ocr.ocr_engine,
        "predict",
        fake_predict,
    )

    result = ocr.ocr_function(
        image,
        scale=2.0,
    )

    assert received == [
        (200, 400, 3),
    ]

    image_space = result[
        "image_space"
    ]

    assert image_space["original"] == {
        "width": 200,
        "height": 100,
        "pixels": 20000,
    }

    assert image_space["processed"] == {
        "width": 400,
        "height": 200,
        "pixels": 80000,
    }

    assert (
        image_space[
            "pixel_area_scale"
        ]
        == pytest.approx(4.0)
    )


@pytest.mark.parametrize(
    "scale, expected_height, "
    "expected_width, expected_area_scale",
    [
        (
            0.5,
            50,
            100,
            0.25,
        ),
        (
            1.0,
            100,
            200,
            1.0,
        ),
        (
            1.5,
            150,
            300,
            2.25,
        ),
        (
            2.0,
            200,
            400,
            4.0,
        ),
    ],
)
def test_image_space_scaling(
    monkeypatch,
    scale,
    expected_height,
    expected_width,
    expected_area_scale,
):
    image = np.zeros(
        (100, 200, 3),
        dtype=np.uint8,
    )

    monkeypatch.setattr(
        ocr.ocr_engine,
        "predict",
        lambda image: [],
    )

    result = ocr.ocr_function(
        image,
        scale=scale,
    )

    processed = result[
        "image_space"
    ]["processed"]

    assert (
        processed["height"]
        == expected_height
    )

    assert (
        processed["width"]
        == expected_width
    )

    assert (
        result[
            "image_space"
        ][
            "pixel_area_scale"
        ]
        == pytest.approx(
            expected_area_scale
        )
    )


# ============================================================
# Multiple PaddleOCR result objects
# ============================================================


def test_ocr_function_combines_multiple_results(
    monkeypatch,
):
    image = np.zeros(
        (100, 200, 3),
        dtype=np.uint8,
    )

    monkeypatch.setattr(
        ocr.ocr_engine,
        "predict",
        lambda image: [
            FakeResult(
                texts=[
                    "OLD TOM DISTILLERY",
                ],
                scores=[
                    0.99,
                ],
                boxes=[
                    [10, 10, 180, 30],
                ],
            ),

            FakeResult(
                texts=[
                    "BOURBON WHISKEY",
                ],
                scores=[
                    0.97,
                ],
                boxes=[
                    [20, 40, 170, 60],
                ],
            ),
        ],
    )

    result = ocr.ocr_function(
        image
    )

    assert result["lines"] == [
        "OLD TOM DISTILLERY",
        "BOURBON WHISKEY",
    ]

    assert result["scores"] == [
        0.99,
        0.97,
    ]

    assert result["boxes"] == [
        [10, 10, 180, 30],
        [20, 40, 170, 60],
    ]

    assert result["text"] == (
        "OLD TOM DISTILLERY\n"
        "BOURBON WHISKEY"
    )


# ============================================================
# Empty PaddleOCR result
# ============================================================


def test_ocr_function_handles_no_results(
    monkeypatch,
):
    image = np.zeros(
        (100, 200, 3),
        dtype=np.uint8,
    )

    monkeypatch.setattr(
        ocr.ocr_engine,
        "predict",
        lambda image: [],
    )

    result = ocr.ocr_function(
        image
    )

    assert result["text"] == ""
    assert result["lines"] == []
    assert result["scores"] == []
    assert result["boxes"] == []


# ============================================================
# Missing OCR result fields
# ============================================================


def test_ocr_function_handles_missing_result_fields(
    monkeypatch,
):
    image = np.zeros(
        (100, 200, 3),
        dtype=np.uint8,
    )

    result_object = FakeResult()

    result_object.json = {
        "res": {}
    }

    monkeypatch.setattr(
        ocr.ocr_engine,
        "predict",
        lambda image: [
            result_object
        ],
    )

    result = ocr.ocr_function(
        image
    )

    assert result["text"] == ""
    assert result["lines"] == []
    assert result["scores"] == []
    assert result["boxes"] == []


# ============================================================
# Score normalization
# ============================================================


def test_ocr_scores_are_converted_to_python_floats(
    monkeypatch,
):
    image = np.zeros(
        (100, 200, 3),
        dtype=np.uint8,
    )

    monkeypatch.setattr(
        ocr.ocr_engine,
        "predict",
        lambda image: [
            FakeResult(
                texts=[
                    "JAMESON",
                ],
                scores=[
                    np.float32(0.95),
                ],
                boxes=[
                    [0, 0, 100, 20],
                ],
            )
        ],
    )

    result = ocr.ocr_function(
        image
    )

    assert isinstance(
        result["scores"][0],
        float,
    )

    assert (
        result["scores"][0]
        == pytest.approx(0.95)
    )


# ============================================================
# Original input should not be resized in place
# ============================================================


def test_scaling_does_not_change_original_image_shape(
    monkeypatch,
):
    image = np.zeros(
        (100, 200, 3),
        dtype=np.uint8,
    )

    original_shape = image.shape

    monkeypatch.setattr(
        ocr.ocr_engine,
        "predict",
        lambda image: [],
    )

    ocr.ocr_function(
        image,
        scale=2.0,
    )

    assert (
        image.shape
        == original_shape
    )