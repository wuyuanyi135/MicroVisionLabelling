from pycocotools import coco
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import json
import sys
import random
import pylab

dataset = coco.COCO(sys.argv[1])

# display COCO categories and supercategories
cats = dataset.loadCats(dataset.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))

image_keys = dataset.imgs.keys()
image_count = len(image_keys)
if len(sys.argv) > 2:
    selected_image_key = sys.argv[2]
else:
    selected_image_key = image_keys[random.randint(0, image_count)]

selected_image = dataset.imgs[selected_image_key]
image_path = selected_image['file_name']

annIds = dataset.getAnnIds(imgIds=selected_image_key, iscrowd=None)
anns = dataset.loadAnns(annIds)
I = io.imread(image_path)
plt.imshow(I); plt.axis('off')
dataset.showAnns(anns)
pylab.show()