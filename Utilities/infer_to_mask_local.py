"""
Use this file to process images and generate the pickle file.
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
from etaprogress.progress import ProgressBar
import pickle
import json
import numpy as np
from pycocotools import coco, mask
import glob
import skimage.measure as measure
import os
import csv
from Queue import Queue
from threading import Thread
import threading

parser = argparse.ArgumentParser()
parser.add_argument("image_path", help="path to the images folder")
parser.add_argument("--output-file", help="pickle output", default="infer_result")
parser.add_argument("--cfg", help="config file path", required=True)
parser.add_argument("--ext", help="image extension", default="png")
parser.add_argument("--wts", help="weight file path", required=True)

args = parser.parse_args()
ext = args.ext
image_path = args.image_path
output_file = args.output_file

wts = args.wts
assert wts

config_path = args.cfg
assert config_path and os.path.exists(config_path)


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


print("Output file: {}".format(output_file))
out = open(output_file, 'wb')
num_files = len(files)
class Loader (Queue):
    def __init__(self, files):
        Queue.__init__(self, 64)
        self.files = files
        self.worker = Thread(target=self.load_img)
        self.worker.setDaemon(True)
        self.worker.start()
        self.finished = False
    
    def load_img(self):
        while True:
            if self.full():
                time.sleep(0)
            else:
                if not len(self.files):
                    self.finished = True
                    break;
                file = self.files.pop(0)
                img = io.imread(file)
                img = color.gray2rgb(img)
                self.put((file, img))
                self.task_done()
loaderQueue = Loader(files)

counter = 0
bar = ProgressBar(num_files, max_width=40)
while not(loaderQueue.finished and loaderQueue.empty()):
    # img = io.imread(file)
    # img = color.gray2rgb(img)
    counter += 1
    bar.numerator = counter 
    print("qsize: {}".format(loaderQueue.qsize()), end=' ')
    print(bar, end='\r')
    sys.stdout.flush()
    file, img = loaderQueue.get()
    with c2_utils.NamedCudaScope(0):
        cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
            model, img, None
        )
    pickle.dump({"f":os.path.basename(file), "m":{"cls_boxes":cls_boxes, "cls_segms":cls_segms}}, out)
