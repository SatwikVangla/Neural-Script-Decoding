from neural_script_decoding import create_app


app = create_app()
ocr_engine = app.extensions["ocr_engine"]


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
