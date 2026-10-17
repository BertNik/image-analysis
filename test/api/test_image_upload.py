import cv2
import numpy as np

from lib.api.image_upload import (
    decode_uploaded_image,
)


def test_decode_uploaded_image():
    source = np.zeros(
        (10, 10, 3),
        dtype=np.uint8,
    )

    success, encoded = cv2.imencode(
        ".png",
        source,
    )

    assert success is True

    result = decode_uploaded_image(
        encoded.tobytes()
    )

    assert result is not None

    assert result.shape == (
        10,
        10,
        3,
    )

def test_decode_uploaded_image_returns_none_for_invalid_bytes():
    result = decode_uploaded_image(
        b"not-an-image"
    )

    assert result is None