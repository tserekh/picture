import os
from pathlib import Path

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

PATH_TO_LABELS = '/home/tserekh/exp/models/research/object_detection/data/mscoco_label_map.pbtxt'
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'

MAX_CONTENT_LENGTH = 1024 * 1024 * 1024
UPLOAD_FOLDER = '../uploads'
RESAVE_FOLDER = '../resaved'

ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4'}
