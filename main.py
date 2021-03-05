import datetime
import os
from logging.config import dictConfig
from pathlib import Path

import numpy as np
import pytz
from PIL import Image
from flask import Flask, request, render_template
from object_detection.utils import label_map_util

from source import config
from source.config import logging_config
from source.ml import load_model, get_class_names_dict, recongnize_image, recongnize_video

app = Flask(__name__)

dictConfig(logging_config)
ROOT_DIR = Path(".")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
model = load_model(config.MODEL_NAME)

def get_extension(filename: str):
    return filename.split('.')[-1].lower()

@app.route('/')
def upload_form():
    class_names_dict = get_class_names_dict(config.PATH_TO_LABELS)
    return render_template('upload.html', class_names=class_names_dict.keys())


@app.route('/', methods=['POST'])
def upload_file():
    class_names_dict = get_class_names_dict(config.PATH_TO_LABELS)
    class_name = request.form.get("class_name")
    file = request.files['file']
    extension = get_extension(file.filename)
    if extension not in config.ALLOWED_IMAGE_EXTENSIONS | config.ALLOWED_VIDEO_EXTENSIONS:
        return render_template(
            'ploblems_with_loading.html',
            extension=extension,
            allowed_extensions=config.ALLOWED_IMAGE_EXTENSIONS | config.ALLOWED_VIDEO_EXTENSIONS)

    os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
    now = datetime.datetime.now(pytz.timezone('Europe/Moscow')).strftime('%Y-%m-%d_%H%M%S')
    file.save(os.path.join(config.UPLOAD_FOLDER, f'{now}.{extension}'))
    category_index = label_map_util.create_category_index_from_labelmap(config.PATH_TO_LABELS, use_display_name=True)
    image_path = os.path.join(config.UPLOAD_FOLDER, f'{now}.{extension}')
    if extension in config.ALLOWED_IMAGE_EXTENSIONS:
        image_resave_path = os.path.join(config.RESAVE_FOLDER, f'{now}.{extension}')
        image_np = np.array(Image.open(image_path))
        summary = recongnize_image(app, model, image_np, class_names_dict, category_index, class_name,
                                   image_resave_path)
        return render_template('response.html', tables=[summary.to_html(classes='scores', index=False)])
    if extension in config.ALLOWED_VIDEO_EXTENSIONS:
        image_resave_path_pattern = os.path.join(config.RESAVE_FOLDER, now + '_{}.' + 'jpg')
        video_path = os.path.join(config.UPLOAD_FOLDER, f'{now}.{extension}')
        summary = recongnize_video(app, model, video_path, class_names_dict, category_index, class_name,
                                   image_resave_path_pattern)
        return render_template('response.html', tables=[summary.to_html(classes='scores', index=False)])


if __name__ == "__main__":
    app.run(port=9999)
