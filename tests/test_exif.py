import base64
import io
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

from mistral_common.image import download_image, maybe_load_image_from_str_or_bytes
from mistral_common.protocol.instruct.chunk import ImageChunk, ImageURLChunk
from mistral_common.tokens.tokenizers.image import image_from_chunk


def _jpeg_with_orientation() -> bytes:
    r"""Return a 200x100 JPEG with EXIF orientation 6 (rotate 90 CW)."""
    img = Image.new("RGB", (200, 100), color="red")
    exif = img.getexif()
    exif[0x0112] = 6
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    return buf.getvalue()


def test_exif_orientation_correction_image_chunk() -> None:
    loaded_img = Image.open(io.BytesIO(_jpeg_with_orientation()))
    chunk = ImageChunk(image=loaded_img)
    processed_img = image_from_chunk(chunk)

    assert processed_img.size == (100, 200)


def test_exif_orientation_correction_base64() -> None:
    loaded = maybe_load_image_from_str_or_bytes(base64.b64encode(_jpeg_with_orientation()).decode("ascii"))

    assert loaded.size == (100, 200)


def test_exif_orientation_correction_bytes() -> None:
    loaded = maybe_load_image_from_str_or_bytes(_jpeg_with_orientation())

    assert loaded.size == (100, 200)


def test_exif_orientation_correction_pil_image() -> None:
    loaded = maybe_load_image_from_str_or_bytes(Image.open(io.BytesIO(_jpeg_with_orientation())))

    assert loaded.size == (100, 200)


def test_exif_orientation_correction_data_url() -> None:
    data = base64.b64encode(_jpeg_with_orientation()).decode("ascii")
    chunk = ImageURLChunk(image_url=f"data:image/jpeg;base64,{data}")
    processed = image_from_chunk(chunk)

    assert processed.size == (100, 200)


def test_exif_orientation_correction_file_url(tmp_path: Path) -> None:
    path = tmp_path / "rotated.jpg"
    path.write_bytes(_jpeg_with_orientation())
    chunk = ImageURLChunk(image_url=path.as_uri())
    processed = image_from_chunk(chunk)

    assert processed.size == (100, 200)


def test_exif_orientation_correction_http() -> None:
    response = Mock(content=_jpeg_with_orientation())
    with patch("mistral_common.image.requests.get", return_value=response):
        loaded = download_image(url="https://example.com/rotated.jpg", timeout=10.0)

    assert loaded.size == (100, 200)
