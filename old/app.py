from flask import Flask, render_template, request, redirect, url_for, flash, send_file
import os
import uuid
import cv2
import pytesseract
from PIL import Image
import numpy as np
import requests
from fpdf import FPDF

app = Flask(__name__)
app.secret_key = "handwriting_recognition_secret_key"
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    preprocessed_path = image_path.replace('.', '_preprocessed.')
    cv2.imwrite(preprocessed_path, dilated)
    return preprocessed_path

def recognize_text(image_path):
    preprocessed_image_path = preprocess_image(image_path)
    custom_config = r'--oem 3 --psm 6 -l eng'
    img = Image.open(preprocessed_image_path)
    text = pytesseract.image_to_string(img, config=custom_config)
    return text

def correct_text_with_llm(text):
    prompt = f"Correct the following sentence for spelling and grammar based on context:\n{text}"
    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={"model": "mistral", "prompt": prompt, "stream": False}
        )
        if response.status_code == 200:
            return response.json()['response'].strip()
        else:
            return "[Error: LLM failed to respond]"
    except Exception as e:
        return f"[Error: Could not connect to LLM - {str(e)}]"

def generate_pdf(text, output_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    lines = text.split('\n')
    for line in lines:
        pdf.multi_cell(0, 10, line)
    pdf.output(output_path)

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
            raw_text = recognize_text(file_path)
            corrected_text = correct_text_with_llm(raw_text)

            # Generate PDF
            pdf_filename = filename.rsplit('.', 1)[0] + ".pdf"
            pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename)
            generate_pdf(corrected_text, pdf_path)

            return render_template('result.html',
                                   image_path=file_path,
                                   raw_text=raw_text,
                                   corrected_text=corrected_text,
                                   pdf_path=pdf_path)
        except Exception as e:
            flash(f'Error processing image: {str(e)}')
            return redirect(request.url)
    else:
        flash('Allowed file types: png, jpg, jpeg, gif, bmp')
        return redirect(request.url)

@app.route('/download_pdf/<filename>')
def download_pdf(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        flash('PDF not found')
        return redirect(url_for('index'))

@app.errorhandler(413)
def too_large(e):
    flash('File too large (max 16MB)')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)

