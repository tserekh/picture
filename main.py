from pathlib import Path
import pytz
from logging.config import dictConfig
import numpy as np
import os
from PIL import Image
import pandas as pd
from flask import Flask, flash, request, redirect, render_template
import datetime
from class_names import name_id_dict
from config import logging_config
from utils import label_map_util, run_inference_for_single_image, load_model
import sys
sys.path.append('/home/tserekh/exp/models/research/')
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

UPLOAD_FOLDER = '../uploads'
RESAVE_FOLDER = '/home/tserekh/resaved'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESAVE_FOLDER'] = RESAVE_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])

dictConfig(logging_config)

ROOT_DIR = Path(".")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
IMAGE_DIR = 'images'

#####################################################
model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
app.logger.warning('start loading model')
model = load_model(model_name)
app.logger.warning('end loading model')
#####################################################

def get_extension(filename):
    return filename.split('.')[-1].lower()


@app.route('/')
def upload_form():
    colours = name_id_dict.keys()
    return render_template('upload.html', colours=colours)


@app.route('/', methods=['POST'])
def upload_file():
    class_id = name_id_dict[request.form.get("colour")]
    app.logger.info(request.form.get("colour"))

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
        flash(file.filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        now = datetime.datetime.now(pytz.timezone('Europe/Moscow')).strftime('%Y-%m-%d_%H%M%S')
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], f'{now}.{extension}'))
        flash('File successfully uploaded')
        path_to_labels = '/home/tserekh/exp/models/research/object_detection/data/mscoco_label_map.pbtxt'
        category_index = label_map_util.create_category_index_from_labelmap(path_to_labels, use_display_name=True)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{now}.{extension}')
        image_np = np.array(Image.open(image_path))
        df_gb = recongnize(model, image_np, category_index, now, extension)
        return render_template('response.html', tables=[df_gb.to_html(classes='scores')])



if __name__ == "__main__":
    app.run()


def recongnize(model, image_np, category_index, now, extension) -> pd.DataFrame:
    output_dict = run_inference_for_single_image(model, image_np)
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=3)
    image_path_resave = os.path.join(app.config['RESAVE_FOLDER'], f'{now}.{extension}')
    Image.fromarray(image_np).save(image_path_resave, format='JPEG')

    df = pd.DataFrame(
        [output_dict['detection_scores'], output_dict['detection_classes']],
        index=['detection_scores', 'detection_classes']
    ).T
    df['class_name'] = df['detection_classes'].apply(lambda x: category_index[x]['name'])
    df_gb = df.groupby('class_name').agg({'detection_scores': ['sum', 'count']})
    df_gb.columns = ['sum', 'count']
    df_gb = df_gb.reset_index()
    return df_gb