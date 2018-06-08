"""
Use this file to get inference from server.
"""
import requests
import json
import numpy as np
from pycocotools import coco, mask
from matplotlib import pyplot as plt
import glob
import skimage.measure as measure
import os
import argparse
import csv

parser = argparse.ArgumentParser()
parser.add_argument("server_address")
parser.add_argument("image_path", help="path to the images folder")
parser.add_argument("--ext", help="image extension", default="png")
parser.add_argument("--tol", help="polygon approximation tolerance", default=1, type=float)


args = parser.parse_args()

server_address = args.server_address
ext = args.ext
image_path = args.image_path
tol = args.tol

csv_path =os.path.join(image_path, "infer.csv")

files = glob.glob(os.path.join(image_path, '*.' + ext))

for file in files:
    print(file)

print("Will process {} file(s) using the server {}".format(len(file), server_address))

main_label = 'crystal form'
labels = ['alpha', 'beta']

csv_file = open(csv_path, 'w', newline='')
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
    with open(file, 'rb') as f:
        ret = requests.post(server_address, f,headers={'Content-Type': 'application/octet-stream'})
    obj = json.loads(ret.content)

    shapes=[]
    shape_classes = []
    cls_segms = obj['cls_segms']
    

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