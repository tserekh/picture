import os
import mrcnn.config
import mrcnn.utils

from config import COCO_MODEL_PATH
def download_pretrained_model():
    if not os.path.exists(COCO_MODEL_PATH):
        mrcnn.utils.download_trained_weights(COCO_MODEL_PATH)
if __name__ == "__main__":
    download_pretrained_model()
