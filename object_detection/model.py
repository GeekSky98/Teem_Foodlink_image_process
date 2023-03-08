import sys
sys.path.append('./*')

import tensorflow as tf
import tarfile
import os
from utils import get_model_detection_function
from object_detection.utils import config_util
from object_detection.builders import model_builder

model_name = 'ssd_resnet152_v1_fpn_1024x1024_coco17_tpu-8'
image_path = './dtection/models/research/object_detection/test_images/image2.jpg'

def tensorflow_api(model_name, first=False):
    if first:
        tar_path = os.path.join('./dtection/zoo/', model_name + '.tar.gz')
        tar_unzip = tarfile.open(tar_path)
        tar_unzip.extractall('./dtection/zoo')

    pipeline_config = os.path.join('dtection/models/research/object_detection/configs/tf2/', model_name + '.config')
    model_dir = 'dtection/models/research/object_detection/test_data/checkpoint/'

    configs = config_util.get_configs_from_pipeline_file(pipeline_config)
    model_config = configs['model']

    detection_model = model_builder.build(model_config=model_config, is_training=False)

    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(model_dir, 'ckpt-0')).expect_partial()

    model = get_model_detection_function(detection_model)

    return model

model = tensorflow_api(model_name)