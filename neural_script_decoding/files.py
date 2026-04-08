import os
import uuid


ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def build_upload_paths(upload_folder, original_name):
    extension = os.path.splitext(original_name)[1].lower()
    filename = f"{uuid.uuid4()}{extension}"
    file_path = os.path.join(upload_folder, filename)
    stem, ext = os.path.splitext(file_path)
    return {
        "original_name": original_name,
        "filename": filename,
        "file_path": file_path,
        "preprocessed_path": f"{stem}_preprocessed{ext}",
        "pdf_path": f"{file_path}.pdf",
    }


def cleanup_files(*paths):
    for path in paths:
        if path and os.path.exists(path):
            os.remove(path)
