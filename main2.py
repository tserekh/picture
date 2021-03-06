from flask import Flask

app = Flask(__name__)


def get_extension(filename: str):
    return filename.split('.')[-1].lower()


@app.route('/')
def upload_form():
    return 'hello'


if __name__ == "__main__":
    app.run(port=9999)
