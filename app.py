from __future__ import annotations

from pathlib import Path
from lib.case.reconcile import reconcile_case
from lib.api.image_upload import decode_uploaded_image

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import (
    CORSMiddleware,
)
from fastapi.responses import (
    FileResponse,
)

from lib.analyze_image import analyze_image
from lib.semantic_runtime import initialize_semantics

BASE_DIR = Path(__file__).resolve().parent
UI_PATH = BASE_DIR / "html/label-review.html"


MODEL_PATH = "./model"
SEMANTIC_MODEL, SEMANTIC_SCHEMA = initialize_semantics(MODEL_PATH)

app = FastAPI(
    title="Label Analysis Service",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8080",
        "http://localhost:8080",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.get(
    "/",
    response_class=FileResponse,
)

def label_review_ui():
    if not UI_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "label-review.html "
                "was not found"
            ),
        )

    return FileResponse(
        path=UI_PATH,
        media_type="text/html",
    )

@app.post("/analyze")
def analyze_upload(
    files: list[
        UploadFile
    ] = File(...),
):
    results = []

    for image_index, file in enumerate(
        files
    ):
        raw = file.file.read()

        if not raw:
            results.append(
                {
                    "image_index":
                        image_index,

                    "filename":
                        file.filename,

                    "status":
                        "ERROR",

                    "usability": {
                        "usable":
                            False,
                    },

                    "reasons": [
                        "empty_upload",
                    ],

                    "error":
                        "Uploaded file is empty",
                }
            )

            continue

        image = decode_uploaded_image(raw)
       
        if image is None:
            results.append(
                {
                    "image_index":
                        image_index,

                    "filename":
                        file.filename,

                    "status":
                        "ERROR",

                    "usability": {
                        "usable":
                            False,
                    },

                    "reasons": [
                        "image_decode_failed",
                    ],

                    "error":
                        "Unable to decode image",
                }
            )

            continue

        result = analyze_image(
            image,
            semantic_model=
                SEMANTIC_MODEL,
            semantic_schema=
                SEMANTIC_SCHEMA,
            brand_required=True,
        )

        result["image_index"] = (
            image_index
        )

        result["filename"] = (
            file.filename
        )

        results.append(
            result
        )

    return reconcile_case(
        results
    )
