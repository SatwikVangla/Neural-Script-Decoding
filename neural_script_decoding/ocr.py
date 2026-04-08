import logging
import os

try:
    import cv2
except ImportError:  # pragma: no cover - dependency presence varies by environment
    cv2 = None

try:
    import pytesseract
except ImportError:  # pragma: no cover - dependency presence varies by environment
    pytesseract = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover - dependency presence varies by environment
    Image = None

try:
    import numpy as np
except ImportError:  # pragma: no cover - dependency presence varies by environment
    np = None

try:
    import easyocr
except ImportError:  # pragma: no cover - dependency presence varies by environment
    easyocr = None

try:
    import torch
except ImportError:  # pragma: no cover - dependency presence varies by environment
    torch = None

try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
except ImportError:  # pragma: no cover - dependency presence varies by environment
    TrOCRProcessor = None
    VisionEncoderDecoderModel = None


logger = logging.getLogger(__name__)


def dependency_available(*modules):
    return all(module is not None for module in modules)


class MultiOCREngine:
    def __init__(self):
        self.engines = {
            "easyocr": None,
            "trocr_processor": None,
            "trocr_model": None,
            "tesseract": True,
        }
        self.initialized = False

    def initialize_engines(self):
        if self.initialized:
            return

        try:
            if dependency_available(easyocr, torch):
                self.engines["easyocr"] = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
                logger.info("EasyOCR initialized successfully")
            else:
                logger.warning("EasyOCR dependencies are not installed")
        except Exception as exc:
            logger.error("Failed to initialize EasyOCR: %s", exc)
            self.engines["easyocr"] = None

        try:
            if dependency_available(torch, TrOCRProcessor, VisionEncoderDecoderModel):
                self.engines["trocr_processor"] = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
                self.engines["trocr_model"] = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
                logger.info("TrOCR initialized successfully")
            else:
                logger.warning("TrOCR dependencies are not installed")
        except Exception as exc:
            logger.error("Failed to initialize TrOCR: %s", exc)
            self.engines["trocr_processor"] = None
            self.engines["trocr_model"] = None

        self.engines["tesseract"] = dependency_available(pytesseract, Image, np, cv2)
        if self.engines["tesseract"]:
            logger.info("Tesseract OCR ready")
        else:
            logger.warning("Tesseract dependencies are not installed")
        self.initialized = True

    def preprocess_image(self, image_path):
        if not dependency_available(cv2, np):
            raise RuntimeError("OpenCV and NumPy are required for preprocessing")

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Unable to read uploaded image")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray)
        blur = cv2.GaussianBlur(denoised, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        kernel = np.ones((2, 2), np.uint8)
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        root, ext = os.path.splitext(image_path)
        preprocessed_path = f"{root}_preprocessed{ext}"
        cv2.imwrite(preprocessed_path, processed)
        return preprocessed_path

    def tesseract_ocr(self, image_path, preprocessed_path=None):
        try:
            if not self.engines["tesseract"]:
                return {"text": "", "confidence": 0, "engine": "tesseract", "error": "Tesseract dependencies not installed"}
            if preprocessed_path is None:
                preprocessed_path = self.preprocess_image(image_path)

            config = r"--oem 3 --psm 6 -l eng"
            img = Image.open(preprocessed_path)
            text = pytesseract.image_to_string(img, config=config)
            confidence = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)
            scores = [float(conf) for conf in confidence["conf"] if float(conf) > 0]
            avg_confidence = np.mean(scores) if scores else 0
            return {"text": text.strip(), "confidence": avg_confidence, "engine": "tesseract"}
        except Exception as exc:
            logger.error("Tesseract OCR failed: %s", exc)
            return {"text": "", "confidence": 0, "engine": "tesseract", "error": str(exc)}

    def easyocr_ocr(self, image_path):
        try:
            if self.engines["easyocr"] is None:
                return {"text": "", "confidence": 0, "engine": "easyocr", "error": "EasyOCR not initialized"}
            results = self.engines["easyocr"].readtext(image_path)
            text_parts = []
            confidences = []
            for _, text, confidence in results:
                text_parts.append(text)
                confidences.append(confidence)
            avg_confidence = np.mean(confidences) * 100 if confidences else 0
            return {"text": " ".join(text_parts), "confidence": avg_confidence, "engine": "easyocr"}
        except Exception as exc:
            logger.error("EasyOCR failed: %s", exc)
            return {"text": "", "confidence": 0, "engine": "easyocr", "error": str(exc)}

    def trocr_ocr(self, image_path):
        try:
            if self.engines["trocr_processor"] is None or self.engines["trocr_model"] is None:
                return {"text": "", "confidence": 0, "engine": "trocr", "error": "TrOCR not initialized"}
            image = Image.open(image_path).convert("RGB")
            pixel_values = self.engines["trocr_processor"](image, return_tensors="pt").pixel_values
            generated_ids = self.engines["trocr_model"].generate(pixel_values)
            generated_text = self.engines["trocr_processor"].batch_decode(generated_ids, skip_special_tokens=True)[0]
            return {"text": generated_text, "confidence": 85, "engine": "trocr"}
        except Exception as exc:
            logger.error("TrOCR failed: %s", exc)
            return {"text": "", "confidence": 0, "engine": "trocr", "error": str(exc)}

    def process_with_all_engines(self, image_path):
        self.initialize_engines()
        results = []
        preprocessed_path = None
        try:
            preprocessed_path = self.preprocess_image(image_path)
        except Exception as exc:
            logger.error("Preprocessing failed: %s", exc)

        results.append(self.tesseract_ocr(image_path, preprocessed_path))
        results.append(self.easyocr_ocr(image_path))
        results.append(self.trocr_ocr(image_path))
        return results

    def combine_results(self, results):
        valid_results = [result for result in results if result["text"] and "error" not in result]
        if not valid_results:
            return "No text could be extracted from the image."

        valid_results.sort(key=lambda item: item["confidence"], reverse=True)
        trocr_result = next((item for item in valid_results if item["engine"] == "trocr"), None)
        if trocr_result and trocr_result["confidence"] > 70:
            logger.info("Using TrOCR result (handwriting optimized)")
            return trocr_result["text"]

        best_result = valid_results[0]
        logger.info("Using %s result (confidence: %.2f)", best_result["engine"], best_result["confidence"])
        return best_result["text"]

    def available_summary(self):
        return {
            "tesseract": bool(self.engines.get("tesseract")),
            "easyocr": self.engines.get("easyocr") is not None,
            "trocr": self.engines.get("trocr_model") is not None,
        }
