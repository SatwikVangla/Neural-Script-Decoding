from pathlib import Path

import pytest

from neural_script_decoding.ocr import MultiOCREngine, cv2, np, pytesseract


pytestmark = pytest.mark.integration


def _tesseract_available():
    if cv2 is None or np is None or pytesseract is None:
        return False
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _tesseract_available(), reason="Tesseract is not installed")
def test_tesseract_processes_generated_fixture(tmp_path):
    from PIL import Image, ImageDraw, ImageFont

    image_path = tmp_path / "fixture.png"
    image = Image.new("RGB", (900, 260), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(Path("static/DejaVuSans.ttf")), 72)
    draw.text((50, 80), "hello ocr", fill="black", font=font)
    image.save(image_path)

    engine = MultiOCREngine(
        {
            "ENABLE_TESSERACT": True,
            "ENABLE_EASYOCR": False,
            "ENABLE_TROCR": False,
        }
    )
    results = engine.process_with_all_engines(str(image_path))
    combined = engine.combine_results(results).lower()

    assert "hello" in combined
    assert "ocr" in combined
