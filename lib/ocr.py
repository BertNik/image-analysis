from paddleocr import PaddleOCR
import paddle
import cv2

ocr_engine = PaddleOCR(
    lang="en",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    text_det_limit_type="max",
    text_det_limit_side_len=1200,
)


def ocr_function(image, scale=1.0):
    h, w = image.shape[:2]

    if scale != 1.0:
        image = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )

    processed_h, processed_w = image.shape[:2]

    results = ocr_engine.predict(image)

    lines = []
    scores = []
    boxes = []

    for result in results:
        data = result.json["res"]

        lines.extend(data.get("rec_texts", []))
        scores.extend(
            float(x)
            for x in data.get("rec_scores", [])
        )
        boxes.extend(data.get("rec_boxes", []))

    return {
        "text": "\n".join(lines),
        "lines": lines,
        "scores": scores,
        "boxes": boxes,

        "image_space": {
            "scale": scale,

            "original": {
                "width": w,
                "height": h,
                "pixels": w * h,
            },

            "processed": {
                "width": processed_w,
                "height": processed_h,
                "pixels": processed_w * processed_h,
            },

            "pixel_area_scale": (
                (processed_w * processed_h)
                / (w * h)
            ),
        },
    }