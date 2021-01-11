from flask import Flask, flash, request, redirect, render_template
import numpy as np
import cv2
import os
import datetime
from class_names import name_id_dict
import pytz
UPLOAD_FOLDER = '../uploads'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'mp4'])
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

dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    }},
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'stream': 'ext://flask.logging.wsgi_errors_stream',
        'formatter': 'default'
    }},
    'root': {
        'level': 'INFO',
        'handlers': ['wsgi']
    }
})
# Configuration that will be used by the Mask-RCNN library
class MaskRCNNConfig(mrcnn.config.Config):
    NAME = "coco_pretrained_model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 80  # COCO dataset has 80 classes + one background class
    DETECTION_MIN_CONFIDENCE = 0.6


# Filter a list of Mask R-CNN detection results to get only the detected cars / trucks
def get_car_boxes(boxes, class_ids, class_id):
    car_boxes = []

    for i, box in enumerate(boxes):
        # If the detected object isn't a car / truck, skip it
        if class_ids[i] in [class_id]:
            car_boxes.append(box)

    return np.array(car_boxes)


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
            cap = cv2.VideoCapture(video_path)

            timestamps = []
            os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], now), exist_ok=True)
            prev_timestamp = -40000
            while True:
                ok, image = cap.read()
                if not ok:
                    break
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
                delta = timestamp - prev_timestamp
                if delta > 20000:

                    timestamps.append(timestamp)
                    rgb_image = image[:, :, ::-1]
                    results = model.detect([rgb_image], verbose=0)
                    r = results[0]
                    car_boxes = get_car_boxes(r['rois'], r['class_ids'], class_id)
                    count = len(car_boxes)
                    # print(int(timestamp / 1000), 'seconds,', count, 'people', end=' ')
                    flash(str(count))
                    plt.figure(figsize=[10, 10])
                    plt.imshow(rgb_image)
                    for box in car_boxes:
                        y1, x1, y2, x2 = box
                        plt.plot([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1], linewidth=3)
                    plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], f'{now}', f'{timestamp}.png'))
                    prev_timestamp = timestamp
            #########################



        return redirect('/')



if __name__ == "__main__":
    app.run()