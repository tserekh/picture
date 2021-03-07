import pathlib
from functools import reduce
from typing import Dict, Any

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops
from object_detection.utils import visualization_utils as vis_util
from tensorflow.python.eager.wrap_function import WrappedFunction


def load_model(model_name: str) -> WrappedFunction:
    base_url = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(
        fname=model_name,
        origin=base_url + model_file,
        untar=True,
    )

    model_dir = pathlib.Path(model_dir) / "saved_model"
    model = tf.saved_model.load(str(model_dir))
    return model


def make_prediction_result(model: WrappedFunction, image: np.array) -> Dict[Any, Any]:
    image = np.asarray(image)
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]
    output_dict = model(input_tensor)

    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    if 'detection_masks' in output_dict:
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    return output_dict


def get_class_names_dict(path_to_labels: str) -> Dict[int, str]:
    category_index = label_map_util.create_category_index_from_labelmap(path_to_labels, use_display_name=True)
    class_names_dict = reduce(lambda x, y: {**x, **y}, map(lambda x: {x['name']: x['id']}, category_index.values()))
    return class_names_dict


def recongnize_image(app, model, image_np, class_names_dict, category_index,
                     class_name, image_path_resave=None) -> pd.DataFrame:
    output_dict = make_prediction_result(model, image_np)
    app.logger.info(str(output_dict))
    app.logger.info(str(class_names_dict))
    app.logger.info(str(class_name))
    filter_ = (output_dict['detection_classes'] == class_names_dict[class_name]) & (
            output_dict['detection_scores'] > 0.5)
    for key in output_dict.keys():
        if key != 'num_detections':
            output_dict[key] = output_dict[key][filter_]
    if image_path_resave is not None:
        save_recognized(image_np, image_path_resave, category_index, output_dict)
    df = pd.DataFrame(
        [output_dict['detection_scores'], output_dict['detection_classes']],
        index=['detection_scores', 'detection_classes']
    ).T
    df['class_name'] = df['detection_classes'].apply(lambda x: {v: k for k, v in class_names_dict.items()}[x])
    summary = df.groupby('class_name').agg({'detection_scores': ['sum', 'count']})
    summary.columns = ['sum', 'count']
    summary = summary.reset_index()
    return summary


def save_recognized(image_np, image_resave_path, category_index, output_dict):
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=3,
        min_score_thresh=0,
    )
    #
    Image.fromarray(image_np).save(image_resave_path, format='JPEG')


def recongnize_video(app, model, video_path, class_names_dict, category_index, class_name,
                     image_resave_path_pattern) -> pd.DataFrame:
    cap = cv2.VideoCapture(video_path)
    brightness = []
    timestamps = []
    summary = pd.DataFrame()
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_preds_for_video = 100
    frames_between_preds = max(1, frame_count // n_preds_for_video)
    j = 0
    while True:
        for i in range(frames_between_preds):
            j += 1
            ok, image_np = cap.read()
            if i != 0:
                continue
            if ok:
                brightness.append(image_np.mean())
                timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
                summary_part = recongnize_image(app, model, image_np, class_names_dict, category_index,
                                                class_name, image_path_resave=image_resave_path_pattern.format(j))
                summary_part['time'] = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
                summary = summary.append(summary_part)
            if not ok:
                break
        if not ok:
            break

    return summary
