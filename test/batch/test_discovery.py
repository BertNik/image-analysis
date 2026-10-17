import pytest

import lib.batch.discovery as discovery

# ============================================================
# find_images
# ============================================================


def test_find_images_recursively_filters_and_sorts(
    tmp_path,
):
    nested = tmp_path / "nested"
    nested.mkdir()

    expected = [
        tmp_path / "a.jpg",
        tmp_path / "b.PNG",
        nested / "c.webp",
    ]

    for path in expected:
        path.write_bytes(b"image")

    ignored = [
        tmp_path / "notes.txt",
        nested / "data.json",
        nested / "no_extension",
    ]

    for path in ignored:
        path.write_text(
            "ignored",
            encoding="utf-8",
        )

    images = discovery.find_images(
        tmp_path
    )

    assert images == sorted(expected)


def test_find_images_returns_empty_list_for_empty_directory(
    tmp_path,
):
    assert (
        discovery.find_images(
            tmp_path
        )
        == []
    )


@pytest.mark.parametrize(
    "extension",
    [
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
        ".PNG",
        ".JPG",
    ],
)
def test_find_images_accepts_supported_extensions(
    tmp_path,
    extension,
):
    path = (
        tmp_path
        / f"label{extension}"
    )

    path.write_bytes(b"image")

    assert (
        discovery.find_images(
            tmp_path
        )
        == [path]
    )
