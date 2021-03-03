import datetime
import os
import sys
from logging.config import dictConfig
from pathlib import Path

import numpy as np
import pytz
from PIL import Image
from flask import Flask, flash, request, redirect, render_template

import config
from config import logging_config
from ml import load_model, get_class_names_dict, recongnize

sys.path.append('/home/tserekh/exp/models/research/')
from object_detection.utils import label_map_util

app = Flask(__name__)
app.secret_key = "secret key"

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])

dictConfig(logging_config)

ROOT_DIR = Path(".")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
IMAGE_DIR = 'images'

#####################################################
app.logger.warning('start loading model')
model = load_model(config.MODEL_NAME)
app.logger.warning('end loading model')
#####################################################

def get_extension(filename):
    return filename.split('.')[-1].lower()


@app.route('/')
def upload_form():
    class_names_dict = get_class_names_dict(config.PATH_TO_LABELS)
    return render_template('upload.html', class_names=class_names_dict.keys())


@app.route('/', methods=['POST'])
def upload_file():
    class_names_dict = get_class_names_dict(config.PATH_TO_LABELS)
    class_name = request.form.get("class_name")
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    flash(file.filename)
    if file.filename == '':
        flash('No file selected for uploading')
        return redirect(request.url)
    extension = get_extension(file.filename)
    os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
    now = datetime.datetime.now(pytz.timezone('Europe/Moscow')).strftime('%Y-%m-%d_%H%M%S')
    file.save(os.path.join(config.UPLOAD_FOLDER, f'{now}.{extension}'))
    image_path = os.path.join(config.UPLOAD_FOLDER, f'{now}.{extension}')
    image_np = np.array(Image.open(image_path))
    category_index = label_map_util.create_category_index_from_labelmap(config.PATH_TO_LABELS, use_display_name=True)

    df_gb = recongnize(app, model, image_np, class_names_dict, category_index, now, extension, config.RESAVE_FOLDER,
                       class_name)
    return render_template('response.html', tables=[df_gb.to_html(classes='scores')])



if __name__ == "__main__":
    app.run()
