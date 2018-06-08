"""
Use this file to get inference locally.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import skimage.io as io
import skimage.color as color
from collections import defaultdict
import argparse
import glob
import logging
import sys
import time
import json
import numpy as np
from pycocotools import coco, mask
import glob
import skimage.measure as measure
import os
import csv


parser = argparse.ArgumentParser()
parser.add_argument("image_path", help="path to the images folder")
parser.add_argument("--cfg", help="config file path", required=True)
parser.add_argument("--ext", help="image extension", default="png")
parser.add_argument("--wts", help="weight file path", required=True)
parser.add_argument("--tol", help="polygon approximation tolerance", default=1, type=float)
parser.add_argument("--label", help="label header for the target object", default="crystal form", type=str)
parser.add_argument("-l", "--label-list", help="label list", action="append")

args = parser.parse_args()

ext = args.ext
image_path = args.image_path
tol = args.tol

wts = args.wts
assert wts

config_path = args.cfg
assert config_path and os.path.exists(config_path)

main_label = args.label
labels = args.label_list

assert isinstance(labels, list)

# load caffe2 and detectron after the arguments are asserted. reducing booting time
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
workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
merge_cfg_from_file(config_path)
cfg.NUM_GPUS = 1
weights = cache_url(wts, cfg.DOWNLOAD_CACHE)
assert_and_infer_cfg(cache_urls=False)
model = infer_engine.initialize_model_from_cfg(weights)

csv_path =os.path.join(image_path, "infer.csv")
files = glob.glob(os.path.join(image_path, '*.' + ext))

print("Will process {} file(s) using detectron".format(len(files)))


try:
    csv_file = open(csv_path, 'w', newline='')
except:
    csv_file = open(csv_path, 'w')
writer = csv.writer(csv_file)
    
writer.writerow([
    "#filename", 
    "file_size", 
    "file_attributes", 
    "region_count",	
    "region_id",
    "region_shape_attributes",
    "region_attributes"
])

for file in files:
    print("Processing: {}".format(file))
    img = io.imread(file)
    img = color.gray2rgb(img)
    with c2_utils.NamedCudaScope(0):
        cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
            model, img, None
        )
    shapes=[]
    shape_classes = []
    

    for i in range(1, len(labels) + 1):
        # processing which class
        current_masks = mask.decode(cls_segms[i])
        for j in range(current_masks.shape[-1]):
            # processing which instance
            current_mask = current_masks[:, :, j]
            contours = measure.find_contours(current_mask, 0.1)
            if (len(contours) > 1):
                print("Warning: segment{},{} contains {} parts. They are seperated into multiple objects".format(i, j, len(contours)))
            for k, contour in enumerate(contours):
                pts = measure.approximate_polygon(contour, tol)

                shapes.append({"name": "polygon", "all_points_x": pts[:,1].tolist(), "all_points_y": pts[:,0].tolist()})
                lbl = {}
                lbl[main_label] = labels[i - 1] # skip the background
                # lbl['segment'] = k
                shape_classes.append(lbl)
    num_objs = len(shapes)

    rows_to_write = [
                    [os.path.basename(file)] * num_objs,
                    [os.path.getsize(file)] * num_objs,
                    ["{}"] * num_objs,
                    [len(shapes)] * num_objs,
                    list(range(num_objs)),
                    [json.dumps(shape) for shape in shapes],
                    [json.dumps(shape_class) for shape_class in shape_classes]
    ]

    # transpose
    rows_to_write = list(map(list, zip(*rows_to_write)))
    writer.writerows(rows_to_write)

print("csv file written to {}".format(csv_path))