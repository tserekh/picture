import os
from pathlib import Path
from object_detection import __file__ as object_detection_path

ROOT_DIR = Path(".")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

logging_config = {
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
}

PATH_TO_LABELS = '/'.join(object_detection_path.split('/')[:-1])+'/data/mscoco_label_map.pbtxt'
MODEL_NAME = 'faster_rcnn_resnet50_v1_640x640_coco17_tpu-8'


MAX_CONTENT_LENGTH = 1024 * 1024 * 1024
UPLOAD_FOLDER = '../uploads'
RESAVE_FOLDER = '../resaved'

ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4'}
