import os
import numpy as np
import cv2
import mrcnn.config
import mrcnn.utils
from mrcnn.model import MaskRCNN
from pathlib import Path
import matplotlib.pyplot as plt
import pytz
from logging.config import dictConfig
from ml import recognize

from flask import Flask, flash, request, redirect, render_template
import datetime
from class_names import name_id_dict
from config import logging_config
UPLOAD_FOLDER = '../uploads'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['mp4'])
from config import MaskRCNNConfig

dictConfig(logging_config)

ROOT_DIR = Path(".")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
if not os.path.exists(COCO_MODEL_PATH):
    mrcnn.utils.download_trained_weights(COCO_MODEL_PATH)

IMAGE_DIR = 'images'


def get_extension(filename):
    return filename.split('.')[-1].lower()


@app.route('/')
def upload_form():
    app.logger.warning('testing warning log')
    app.logger.error('testing error log')
    app.logger.info('testing info log')
    colours = name_id_dict.keys()
    return render_template('upload.html', colours=colours)




@app.route('/', methods=['POST'])
def upload_file():
    class_id = name_id_dict[request.form.get("colour")]
    app.logger.info(request.form.get("colour"))

    model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=MaskRCNNConfig())
    model.load_weights("mask_rcnn_coco.h5", by_name=True)
    flash('model loaded')
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        flash(file.filename)
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        extension = get_extension(file.filename)
        flash(extension)
        if file and (extension in ALLOWED_EXTENSIONS):
            flash(file.filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            now = datetime.datetime.now(pytz.timezone('Europe/Moscow')).strftime('%Y-%m-%d_%H%M%S')


            file.save(os.path.join(app.config['UPLOAD_FOLDER'], f'{now}.{extension}'))
            flash('File successfully uploaded')

            video_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{now}.{extension}')
            step = 3000
            working_folder = os.path.join(app.config['UPLOAD_FOLDER'], f'{now}')
            recognize(video_path, model, class_id, step, working_folder)
            #########################

        return redirect('/')



if __name__ == "__main__":
    app.run()