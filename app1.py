from flask import Flask, render_template, request, redirect, url_for, flash, send_file, jsonify
import os
import uuid
import cv2
import pytesseract
from PIL import Image
import numpy as np
import requests
from fpdf import FPDF
from fpdf.enums import XPos, YPos
import textwrap
import easyocr
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import logging
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = "handwriting_recognition_secret_key"
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class MultiOCREngine:
    def __init__(self):
        self.engines = {}
        self.initialize_engines()
    
    def initialize_engines(self):
        """Initialize all OCR engines"""
        try:
            # Initialize EasyOCR
            self.engines['easyocr'] = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
            logger.info("EasyOCR initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            self.engines['easyocr'] = None
        
        try:
            # Initialize TrOCR for handwriting
            self.engines['trocr_processor'] = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
            self.engines['trocr_model'] = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
            logger.info("TrOCR initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize TrOCR: {e}")
            self.engines['trocr_processor'] = None
            self.engines['trocr_model'] = None
        
        # Tesseract is always available (assumed to be installed)
        self.engines['tesseract'] = True
        logger.info("Tesseract OCR ready")

    def tesseract_ocr(self, image_path, preprocessed_path=None):
        """Tesseract OCR with preprocessing"""
        try:
            if preprocessed_path is None:
                preprocessed_path = self.preprocess_image(image_path)
            
            config = r'--oem 3 --psm 6 -l eng'
            img = Image.open(preprocessed_path)
            text = pytesseract.image_to_string(img, config=config)
            confidence = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)
            avg_confidence = np.mean([int(conf) for conf in confidence['conf'] if int(conf) > 0])
            
            return {
                'text': text.strip(),
                'confidence': avg_confidence,
                'engine': 'tesseract'
            }
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            return {'text': '', 'confidence': 0, 'engine': 'tesseract', 'error': str(e)}

    def easyocr_ocr(self, image_path):
        """EasyOCR processing"""
        try:
            if self.engines['easyocr'] is None:
                return {'text': '', 'confidence': 0, 'engine': 'easyocr', 'error': 'EasyOCR not initialized'}
            
            results = self.engines['easyocr'].readtext(image_path)
            text_parts = []
            confidences = []
            
            for (bbox, text, confidence) in results:
                text_parts.append(text)
                confidences.append(confidence)
            
            combined_text = ' '.join(text_parts)
            avg_confidence = np.mean(confidences) * 100 if confidences else 0
            
            return {
                'text': combined_text,
                'confidence': avg_confidence,
                'engine': 'easyocr'
            }
        except Exception as e:
            logger.error(f"EasyOCR failed: {e}")
            return {'text': '', 'confidence': 0, 'engine': 'easyocr', 'error': str(e)}

    def trocr_ocr(self, image_path):
        """TrOCR processing for handwritten text"""
        try:
            if self.engines['trocr_processor'] is None or self.engines['trocr_model'] is None:
                return {'text': '', 'confidence': 0, 'engine': 'trocr', 'error': 'TrOCR not initialized'}
            
            image = Image.open(image_path).convert('RGB')
            pixel_values = self.engines['trocr_processor'](image, return_tensors="pt").pixel_values
            generated_ids = self.engines['trocr_model'].generate(pixel_values)
            generated_text = self.engines['trocr_processor'].batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return {
                'text': generated_text,
                'confidence': 85,  # TrOCR doesn't provide confidence, using estimated value
                'engine': 'trocr'
            }
        except Exception as e:
            logger.error(f"TrOCR failed: {e}")
            return {'text': '', 'confidence': 0, 'engine': 'trocr', 'error': str(e)}

    def preprocess_image(self, image_path):
        """Enhanced image preprocessing"""
        try:
            img = cv2.imread(image_path)
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Noise reduction
            denoised = cv2.fastNlMeansDenoising(gray)
            
            # Gaussian blur
            blur = cv2.GaussianBlur(denoised, (5, 5), 0)
            
            # Adaptive thresholding
            thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY, 11, 2)
            
            # Morphological operations
            kernel = np.ones((2, 2), np.uint8)
            processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # Save preprocessed image
            preprocessed_path = image_path.replace('.', '_preprocessed.')
            cv2.imwrite(preprocessed_path, processed)
            
            return preprocessed_path
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            raise e

    def process_with_all_engines(self, image_path):
        """Process image with all available OCR engines"""
        results = []
        preprocessed_path = None
        
        try:
            preprocessed_path = self.preprocess_image(image_path)
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
        
        # Run Tesseract
        tesseract_result = self.tesseract_ocr(image_path, preprocessed_path)
        results.append(tesseract_result)
        
        # Run EasyOCR
        easyocr_result = self.easyocr_ocr(image_path)
        results.append(easyocr_result)
        
        # Run TrOCR
        trocr_result = self.trocr_ocr(image_path)
        results.append(trocr_result)
        
        return results

    def combine_results(self, results):
        """Combine results from multiple OCR engines"""
        valid_results = [r for r in results if r['text'] and 'error' not in r]
        
        if not valid_results:
            return "No text could be extracted from the image."
        
        # Sort by confidence
        valid_results.sort(key=lambda x: x['confidence'], reverse=True)
        
        # If TrOCR has reasonable confidence, prefer it for handwriting
        trocr_result = next((r for r in valid_results if r['engine'] == 'trocr'), None)
        if trocr_result and trocr_result['confidence'] > 70:
            logger.info("Using TrOCR result (handwriting optimized)")
            return trocr_result['text']
        
        # Otherwise, use the highest confidence result
        best_result = valid_results[0]
        logger.info(f"Using {best_result['engine']} result (confidence: {best_result['confidence']:.2f})")
        
        return best_result['text']

# Initialize the multi-OCR engine
ocr_engine = MultiOCREngine()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def correct_text_with_llm(text):
    """Correct text using local LLM"""
    prompt = f"""Please correct the following text for spelling and grammar errors. 
    This text was extracted from a handwritten document using OCR, so there may be character recognition errors.
    Return only the corrected text without any explanations:

    {text}"""
    
    try:
        res = requests.post('http://localhost:11434/api/generate',
                            json={"model": "mistral", "prompt": prompt, "stream": False},
                            timeout=30)
        if res.status_code == 200:
            return res.json()['response'].strip()
        return f"[Error: LLM response failed - Status {res.status_code}]"
    except requests.exceptions.RequestException as e:
        logger.error(f"LLM request failed: {e}")
        return f"[Error: LLM offline - {str(e)}]"

def generate_pdf(text, output_path):
    """Generate PDF from text"""
    try:
        pdf = FPDF()
        pdf.add_page()

        # Try to load Unicode font
        font_path = os.path.join("static", "DejaVuSans.ttf")
        if os.path.exists(font_path):
            pdf.add_font("DejaVu", "", font_path, uni=True)
            pdf.set_font("DejaVu", size=12)
        else:
            pdf.set_font("Arial", size=12)

        # Add metadata
        pdf.set_title("OCR Extracted Text")
        pdf.set_author("Multi-OCR App")
        pdf.set_creator("Handwriting Recognition System")

        # Wrap and add text
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                pdf.ln(5)
                continue

            wrapped_lines = textwrap.wrap(line, width=80)
            for subline in wrapped_lines:
                pdf.cell(0, 10, subline, ln=True)

        pdf.output(output_path)
        logger.info(f"PDF generated: {output_path}")
    except Exception as e:
        logger.error(f"PDF generation failed: {e}")
        raise e

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            # Process with all OCR engines
            logger.info(f"Processing image: {filename}")
            ocr_results = ocr_engine.process_with_all_engines(file_path)
            
            # Combine results
            raw_text = ocr_engine.combine_results(ocr_results)
            
            # Correct with LLM
            corrected_text = correct_text_with_llm(raw_text)
            
            # Generate PDF
            output_pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename + ".pdf")
            generate_pdf(corrected_text, output_pdf_path)
            
            # Prepare results for display
            results_data = {
                'ocr_results': ocr_results,
                'raw_text': raw_text,
                'corrected_text': corrected_text,
                'pdf_path': output_pdf_path
            }
            
            return render_template("results.html", **results_data)
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            flash(f"Error processing image: {str(e)}")
            return redirect(request.url)

    flash('Allowed file types are png, jpg, jpeg, gif, bmp')
    return redirect(request.url)

@app.route('/download/<filename>')
def download_pdf(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        flash('File not found')
        return redirect(url_for('index'))

@app.route('/api/ocr', methods=['POST'])
def api_ocr():
    """API endpoint for OCR processing"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Process with all engines
        ocr_results = ocr_engine.process_with_all_engines(file_path)
        combined_text = ocr_engine.combine_results(ocr_results)
        corrected_text = correct_text_with_llm(combined_text)
        
        # Clean up
        os.remove(file_path)
        
        return jsonify({
            'success': True,
            'ocr_results': ocr_results,
            'raw_text': combined_text,
            'corrected_text': corrected_text
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'engines': {
            'tesseract': True,
            'easyocr': ocr_engine.engines['easyocr'] is not None,
            'trocr': ocr_engine.engines['trocr_model'] is not None
        },
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
