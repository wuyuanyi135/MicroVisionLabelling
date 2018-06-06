# create masked dataset
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import skimage.io
import csv
import itertools
import glob
from pycococreatortools import pycococreatortools as tools
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-k', '--key-name', help='key name. leave blank if you want everything to have ID=1')
parser.add_argument('-l','--label-name', action='append', help='label name in order that will map to the IDs')

args = parser.parse_args()
key_name = args.key_name
labels = args.label_name

def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

img_list = []
an_list = []
count = 0

dataset_dir =  os.path.join(os.path.dirname(__file__), '../Dataset/')
dataset_dir = os.path.abspath(dataset_dir) # python2 __file__ is not absolute
batches = [name for name in os.listdir(dataset_dir)
    if os.path.isdir(os.path.join(dataset_dir, name))]
batches_full_path = [os.path.join(dataset_dir, x) for x in batches]

for batch_path in batches_full_path:
    # annotation_file_path = os.path.join(batch_path, "via_region_data.csv")
    # if not os.path.exists(annotation_file_path):
    #     # this batch has not been labelled
    #     continue

    annotation_file_paths = glob.glob(os.path.join(batch_path, "*.csv"));
    for annotation_file_path in annotation_file_paths:


        with open(annotation_file_path, "r") as annotation_file:
            annotation = csv.reader(annotation_file)
            next(annotation) # skip the first comment line
            grouped_annotation = itertools.groupby(annotation, lambda x: x[0])
            for image_file, g in grouped_annotation:
                
                shape_entries = []
                attrs = []
                for x in g:
                    shape_entries.append(json.loads(x[5]));
                    attrs.append(json.loads(x[6]))
                # skip empty files (without label)
                if not bool(shape_entries[0]):
                    continue

                # add image
                image_full_path = os.path.join(batch_path, image_file)
                image = skimage.io.imread(image_full_path)
                height, width = image.shape[:2]
                
                

                for i, p in enumerate(shape_entries):
                    try:
                        attr = attrs[i]
                    except Exception:
                        print ("Num attr is not matching Num shapes. Please check the annotation files")
                        exit(-1)

                    if (not attr):
                        continue
                    mask = np.zeros([height, width],
                            dtype=np.uint8)
                    # Get indexes of pixels inside the polygon and set them to 1
                    if p['name'] == 'polygon':
                        rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
                        mask[rr, cc] = 1

                    if p['name'] == 'circle':
                        rr, cc = skimage.draw.circle(p['cy'], p['cx'], p['r'], (height, width))
                        mask[rr, cc] = 1
                    
                    if key_name:
                        try:
                            id = labels.index(attr[key_name]) + 1
                        except:
                            print ("{} is not in labels: {}".format(attr[key_name], labels))
                            exit(-1)
                    else:
                        id = 1

                    ann = tools.create_annotation_info(image_file+str(i), image_file, {"id": id, 'is_crowd': False}, mask, (width, height), tolerance=0.1)
                    if not ann:
                        continue
                    an_list.append(ann)
                    count = count + 1
                img_list.append(tools.create_image_info(image_file, image_full_path, (width, height)))


INFO = {
    "description": "Crystal Dataset",
    "url": "https://github.com/wuyuanyi135/",
    "version": "0.1.0",
    "year": 2018,
    "contributor": "wuyuanyi",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}
LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = []
if key_name:
    for i, label in enumerate(labels):
        CATEGORIES.append({"id": i+1, "name": label, "supercategory": "object"})
else:
    CATEGORIES.append({"id": 1, "name": "object", "supercategory": "object"})
coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": img_list,
        "annotations": an_list
    }

with open('instances_train2014.json', 'w') as outfile:
    json.dump(coco_output, outfile)
print("{} objects in dataset".format(count))

