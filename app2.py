from flask import Flask, render_template, request, redirect, url_for, flash, send_file
import os
import uuid
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from fpdf import FPDF
import textwrap
import requests

app = Flask(__name__)
app.secret_key = "handwriting_recognition_secret_key"
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load TrOCR
print("Loading TrOCR...")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-stage1")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("TrOCR loaded.")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def recognize_text_with_trocr(image_path):
    image = Image.open(image_path).convert("RGB")

    # Use full-size image for maximum accuracy
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values,
            max_length=256,
            num_beams=4,
            early_stopping=True
        )
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text.strip()

def correct_text_with_llm(text):
    prompt = f"Correct the following sentence for spelling and grammar based on context and give me only the corrected paragraph:\n{text}"
    try:
        res = requests.post('http://localhost:11434/api/generate',
                            json={"model": "mistral", "prompt": prompt, "stream": False})
        if res.status_code == 200:
            return res.json()['response'].strip()
        return "[Error: LLM response failed]"
    except Exception as e:
        return f"[Error: LLM offline - {str(e)}]"

def generate_pdf(text, output_path):
    pdf = FPDF()
    pdf.add_page()
    font_path = os.path.join("static", "DejaVuSans.ttf")
    if os.path.exists(font_path):
        pdf.add_font("DejaVu", "", font_path, uni=True)
        pdf.set_font("DejaVu", size=12)
    else:
        pdf.set_font("Arial", size=12)

    for line in text.split('\n'):
        line = line.strip()
        if not line:
            pdf.ln(5)
            continue
        wrapped = textwrap.wrap(line, width=80)
        for subline in wrapped:
            pdf.cell(0, 10, subline, ln=True)

    pdf.output(output_path)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'GET':
        return redirect(url_for('index'))

    if 'file' not in request.files:
        flash("No file part")
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        flash("No selected file")
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            raw_text = recognize_text_with_trocr(filepath)
            print("📝 Raw OCR Output:", raw_text)

            # If OCR failed or is not meaningful, skip LLM
            if len(raw_text.split()) < 5:
                corrected_text = raw_text
                print("⚠️ Skipping LLM correction (too short or noisy)")
            else:
                corrected_text = correct_text_with_llm(raw_text)
                print("✅ LLM Corrected:", corrected_text)

            output_pdf_path = filepath + ".pdf"
            generate_pdf(corrected_text, output_pdf_path)
            return render_template("download.html", pdf_path=output_pdf_path)

        except Exception as e:
            print(f"❌ Processing error: {e}")
            flash(f"Error: {e}")
            return redirect(request.url)

    flash("Allowed file types: png, jpg, jpeg")
    return redirect(request.url)

@app.route('/download/<filename>')
def download_pdf(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)

