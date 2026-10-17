
from __future__ import annotations

import cv2
import numpy as np


def decode_uploaded_image(
    raw: bytes,
):
    if not raw:
        return None

    encoded = np.frombuffer(
        raw,
        dtype=np.uint8,
    )

    return cv2.imdecode(
        encoded,
        cv2.IMREAD_COLOR,
    )
