import os
import pathlib
import sys
from functools import reduce
from typing import Dict, Any

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tensorflow.python.eager.wrap_function import WrappedFunction

sys.path.append('/home/tserekh/exp/models/research/')
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


def load_model(model_name: str) -> WrappedFunction:
    base_url = 'http://download.tensorflow.org/models/object_detection/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(
        fname=model_name,
        origin=base_url + model_file,
        untar=True)

    model_dir = pathlib.Path(model_dir) / "saved_model"
    model = tf.saved_model.load(str(model_dir))
    model = model.signatures['serving_default']
    return model


def make_prediction_result(model: WrappedFunction, image: np.array) -> Dict[Any]:
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


def recongnize(app, model, image_np, class_names_dict, category_index, now, extension, resave_folder,
               class_name) -> pd.DataFrame:
    output_dict = make_prediction_result(model, image_np)

    app.logger.warning(str(output_dict))
    app.logger.warning(str(class_names_dict))
    app.logger.warning(str(class_name))
    filter_ = (output_dict['detection_classes'] == class_names_dict[class_name]) & (
            output_dict['detection_scores'] > 0.5)
    for key in output_dict.keys():
        if key != 'num_detections':
            output_dict[key] = output_dict[key][filter_]
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
    image_path_resave = os.path.join(resave_folder, f'{now}.{extension}')
    Image.fromarray(image_np).save(image_path_resave, format='JPEG')

    df = pd.DataFrame(
        [output_dict['detection_scores'], output_dict['detection_classes']],
        index=['detection_scores', 'detection_classes']
    ).T
    df['class_name'] = df['detection_classes'].apply(lambda x: {v: k for k, v in class_names_dict.items()}[x])
    df_gb = df.groupby('class_name').agg({'detection_scores': ['sum', 'count']})
    df_gb.columns = ['sum', 'count']
    df_gb = df_gb.reset_index()
    return df_gb
