from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, abort
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

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(32))
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

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
    root, ext = os.path.splitext(image_path)
    preprocessed_path = f"{root}_preprocessed{ext}"
    cv2.imwrite(preprocessed_path, dilated)
    return preprocessed_path

def cleanup_files(*paths):
    for path in paths:
        if path and os.path.exists(path):
            os.remove(path)

def preprocessed_path_for(image_path):
    root, ext = os.path.splitext(image_path)
    return f"{root}_preprocessed{ext}"

def recognize_text(image_path):
    preprocessed_image_path = preprocess_image(image_path)
    config = r'--oem 3 --psm 6 -l eng'
    img = Image.open(preprocessed_image_path)
    return pytesseract.image_to_string(img, config=config)

def correct_text_with_llm(text):
    prompt = f"Correct the following sentence for spelling and grammar based on context and give me only the corrected paragraph and don't give me the description of what words you have corrected:\n{text}"
    try:
        res = requests.post('http://localhost:11434/api/generate',
                            json={"model": "mistral", "prompt": prompt, "stream": False},
                            timeout=30)
        if res.status_code == 200:
            return res.json()['response'].strip()
        return "[Error: LLM response failed]"
    except Exception as e:
        return f"[Error: LLM offline - {str(e)}]"

def generate_pdf(text, output_path):
    pdf = FPDF()
    pdf.add_page()

    # Load Unicode font
    font_path = os.path.join("static", "DejaVuSans.ttf")
    pdf.add_font("DejaVu", "", font_path, uni=True)
    pdf.set_font("DejaVu", size=12)

    # Wrap each line to avoid long unbreakable chunks
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            pdf.ln(5)
            continue

        # Wrap long lines manually
        wrapped_lines = textwrap.wrap(line, width=80)  # 80 chars max per line
        for subline in wrapped_lines:
            pdf.cell(0, 10, subline, ln=True)

    pdf.output(output_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        preprocessed_path = preprocessed_path_for(file_path)
        file.save(file_path)

        try:
            raw_text = recognize_text(file_path)
            corrected_text = correct_text_with_llm(raw_text)  # ← LLM here
            output_pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename + ".pdf")
            generate_pdf(corrected_text, output_pdf_path)
            cleanup_files(file_path, preprocessed_path)
            return render_template("download.html", pdf_path=output_pdf_path)
        except Exception as e:
            flash(f"Error: {str(e)}")
            cleanup_files(file_path, preprocessed_path)
            return redirect(url_for('index'))

    flash('Allowed file types are png, jpg, jpeg, gif, bmp')
    return redirect(url_for('index'))

@app.route('/download/<filename>')
def download_pdf(filename):
    safe_name = os.path.basename(filename)
    if safe_name != filename:
        abort(404)
    return send_from_directory(app.config['UPLOAD_FOLDER'], safe_name, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
