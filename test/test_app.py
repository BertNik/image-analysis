import numpy as np
import pytest
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse

from io import BytesIO

from starlette.datastructures import (
    UploadFile,
)
from fastapi.testclient import (
    TestClient,
)
from test.helpers.case_fixture import complete_image
import app as app

def test_label_review_returns_404_when_ui_missing(
    monkeypatch,
    tmp_path,
):
    missing_path = (
        tmp_path
        / "label-review.html"
    )

    assert not missing_path.exists()

    monkeypatch.setattr(
        app,
        "UI_PATH",
        missing_path,
    )

    with pytest.raises(
        HTTPException
    ) as captured:
        app.label_review_ui()

    error = captured.value

    assert error.status_code == 404

    assert error.detail == (
        "label-review.html "
        "was not found"
    )
    
def test_label_review_returns_html_file(
    monkeypatch,
    tmp_path,
):
    ui_path = (
        tmp_path
        / "label-review.html"
    )

    ui_path.write_text(
        "<!doctype html>"
        "<html><body>Label Review</body></html>",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        app,
        "UI_PATH",
        ui_path,
    )

    response = app.label_review_ui()

    assert isinstance(
        response,
        FileResponse,
    )

    assert Path(
        response.path
    ) == ui_path

    assert response.media_type == (
        "text/html"
    )

def test_empty_upload_abstains_case():
    file = UploadFile(
        filename="empty.png",
        file=BytesIO(b""),
    )

    result = app.analyze_upload(
        files=[file],
    )

    assert result["status"] == "ABSTAIN"

    assert result["reasons"] == [
        "no_usable_image_evidence",
    ]

    assert len(
        result["images"]
    ) == 1

    image = result["images"][0]

    assert image["image_index"] == 0

    assert image["filename"] == (
        "empty.png"
    )

    assert image["status"] == "ERROR"

    assert image["error"] == (
        "Uploaded file is empty"
    )

    assert result["analysis"][
        "analysis_complete"
    ] is False


def test_empty_upload_builds_expected_image_error(
    monkeypatch,
):
    file = UploadFile(
        filename="empty.png",
        file=BytesIO(b""),
    )

    captured = {}

    def fake_reconcile_case(
        image_results,
    ):
        captured["results"] = (
            image_results
        )

        return {
            "captured": True,
        }

    monkeypatch.setattr(
        app,
        "reconcile_case",
        fake_reconcile_case,
    )

    result = app.analyze_upload(
        files=[file],
    )

    assert result == {
        "captured": True,
    }

    assert len(
        captured["results"]
    ) == 1

    image = captured[
        "results"
    ][0]

    assert image == {
        "image_index": 0,
        "filename": "empty.png",
        "status": "ERROR",
        "usability": {
            "usable": False,
        },
        "reasons": [
            "empty_upload",
        ],
        "error":
            "Uploaded file is empty",
    }


def test_empty_upload_does_not_run_analysis(
    monkeypatch,
):
    file = UploadFile(
        filename="empty.png",
        file=BytesIO(b""),
    )

    def analysis_should_not_run(
        *args,
        **kwargs,
    ):
        raise AssertionError(
            "analyze_image must not run "
            "for an empty upload"
        )

    monkeypatch.setattr(
        app,
        "analyze_image",
        analysis_should_not_run,
    )

    result = app.analyze_upload(
        files=[file],
    )

    assert result["status"] == (
        "ABSTAIN"
    )

    assert result["reasons"] == [
        "no_usable_image_evidence",
    ]


def test_analyze_endpoint_single_image(
    monkeypatch,
):
    fake_image = np.zeros(
        (10, 10, 3),
        dtype=np.uint8,
    )

    monkeypatch.setattr(
        app,
        "decode_uploaded_image",
        lambda raw: fake_image,
    )

    captured = {}

    def fake_analyze_image(
        image,
        *,
        semantic_model=None,
        semantic_schema=None,
        brand_required=False,
    ):
        captured["image"] = image
        captured["semantic_model"] = (
            semantic_model
        )
        captured["semantic_schema"] = (
            semantic_schema
        )
        captured["brand_required"] = (
            brand_required
        )

        return complete_image()

    monkeypatch.setattr(
        app,
        "analyze_image",
        fake_analyze_image,
    )

    client = TestClient(
        app.app
    )

    response = client.post(
        "/analyze",
        files=[
            (
                "files",
                (
                    "label.jpg",
                    b"fake-image-data",
                    "image/jpeg",
                ),
            )
        ],
    )

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == (
        "RESOLVED"
    )

    assert (
        captured["image"]
        is fake_image
    )

    assert (
        captured["semantic_model"]
        is app.SEMANTIC_MODEL
    )

    assert (
        captured["semantic_schema"]
        is app.SEMANTIC_SCHEMA
    )

    assert (
        captured["brand_required"]
        is True
    )

def test_analyze_endpoint_bad_image_abstains(
    monkeypatch,
):
    monkeypatch.setattr(
        app,
        "decode_uploaded_image",
        lambda raw: None,
    )

    from fastapi.testclient import (
        TestClient,
    )

    client = TestClient(
        app.app
    )

    response = client.post(
        "/analyze",
        files=[
            (
                "files",
                (
                    "bad.jpg",
                    b"not-an-image",
                    "image/jpeg",
                ),
            )
        ],
    )

    assert (
        response.status_code
        == 200
    )

    body = response.json()

    assert (
        body["status"]
        == "ABSTAIN"
    )

    assert body["reasons"] == [
        "no_usable_image_evidence"
    ]
