
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from flask import Flask, request, jsonify
import cStringIO
import skimage.io as io
import skimage.color as color


from collections import defaultdict
import argparse
import glob
import logging
import os
import sys
import time

from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils

c2_utils.import_detectron_ops()

app = Flask(__name__)
workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
merge_cfg_from_file('/home/wuyuanyi/Desktop/Detectron/configs/fine_tuning/e2e_mask_rcnn_R-50-FPN_1x.yaml')
cfg.NUM_GPUS = 1
weights = cache_url('/home/wuyuanyi/Desktop/GluTraining/train/coco_2014_train/generalized_rcnn/model_final.pkl', cfg.DOWNLOAD_CACHE)
assert_and_infer_cfg(cache_urls=False)
model = infer_engine.initialize_model_from_cfg(weights)


@app.route('/', methods=['POST'])
def process_image():
    img_stringIO = cStringIO.StringIO(request.data)
    img = io.imread(img_stringIO)
    img = color.gray2rgb(img)
    with c2_utils.NamedCudaScope(0):
        cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
            model, img, None
        )
    cls_box_serializable = []
    for cls_box in cls_boxes: 
        if 'tolist' in dir(cls_box):
            cls_box_serializable.append(cls_box.tolist())
        else:
            cls_box_serializable.append(cls_box)
    return jsonify({'cls_boxes': cls_box_serializable, 'cls_segms': cls_segms})

app.run(debug=True, port=9088, host='0.0.0.0')