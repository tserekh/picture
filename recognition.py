import os
import numpy as np
import cv2
import mrcnn.utils
from mrcnn.model import MaskRCNN
from pathlib import Path

import glob


# Configuration that will be used by the Mask-RCNN library
class MaskRCNNConfig(mrcnn.config.Config):
    NAME = "coco_pretrained_model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 80  # COCO dataset has 80 classes + one background class
    DETECTION_MIN_CONFIDENCE = 0.6


# Filter a list of Mask R-CNN detection results to get only the detected cars / trucks
def get_car_boxes(boxes, class_ids):
    car_boxes = []

    for i, box in enumerate(boxes):
        # If the detected object isn't a car / truck, skip it
        if class_ids[i] in [1]:
            car_boxes.append(box)

    return np.array(car_boxes)


# Root directory of the project
ROOT_DIR = Path(".")

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    mrcnn.utils.download_trained_weights(COCO_MODEL_PATH)

IMAGE_DIR = 'images'


def calc_new_num(file_path):
    model = MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=MaskRCNNConfig())
    model.load_weights(COCO_MODEL_PATH, by_name=True)
    image = cv2.imread(file_path)
    rgb_image = image[:, :, ::-1]
    results = model.detect([rgb_image], verbose=0)
    #     try:
    #         results = model.detect([rgb_image], verbose=0)
    #     except InvalidArgumentError:
    #         print('exception')
    #         continue
    r = results[0]
    car_boxes = get_car_boxes(r['rois'], r['class_ids'])

    f = open('num.txt', mode='w')
    f.write(str(len(car_boxes)))
    f.close()


#     plt.figure(figsize=[10,10])
#     plt.plot()
#     plt.imshow(rgb_image)
#     for box in car_boxes:
#         y1, x1, y2, x2 = box
#         plt.plot([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1],linewidth=3)

#     plt.show()

file_path = sorted(glob.glob('incoming_images/*'))[-1]
calc_new_num(file_path)