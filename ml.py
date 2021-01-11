import os
import numpy as np
import cv2
import mrcnn.config
import mrcnn.utils
import matplotlib.pyplot as plt
from flask import flash

class MaskRCNNConfig(mrcnn.config.Config):
    NAME = "coco_pretrained_model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 80  # COCO dataset has 80 classes + one background class
    DETECTION_MIN_CONFIDENCE = 0.6

IMAGE_DIR = 'images'
def select_boxes(all_boxes, class_ids, class_id):
    boxes = []
    for i, box in enumerate(all_boxes):
        if class_ids[i] in [class_id]:
            boxes.append(box)
    return np.array(boxes)

def recognize(video_path, model, class_id, step, working_folder):
    cap = cv2.VideoCapture(video_path)
    timestamps = []
    os.makedirs(working_folder, exist_ok=True)
    prev_timestamp = step*2
    while True:
        ok, image = cap.read()
        if not ok:
            break
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
        delta = timestamp - prev_timestamp
        if delta > step:
            timestamps.append(timestamp)
            rgb_image = image[:, :, ::-1]
            results = model.detect([rgb_image], verbose=0)
            r = results[0]
            selected_boxes = select_boxes(r['rois'], r['class_ids'], class_id)
            count = len(selected_boxes)
            flash(str(count))
            plt.figure(figsize=[10, 10])
            plt.imshow(rgb_image)
            for box in selected_boxes:
                y1, x1, y2, x2 = box
                plt.plot([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1], linewidth=3)
            plt.savefig(os.path.join(working_folder, f'{timestamp}.png'))
            prev_timestamp = timestamp