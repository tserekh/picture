from flask import Flask
app = Flask(__name__)
from waitress import serve


@app.route('/')
def hello_world():
    return 'Hello, World!1111'

if __name__ == '__main__':
    serve(app, port=8000)