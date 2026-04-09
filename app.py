from neural_script_decoding import create_app


app = create_app()


if __name__ == "__main__":
    app.run(
        debug=app.config["APP_DEBUG"],
        host=app.config["APP_HOST"],
        port=app.config["APP_PORT"],
    )
