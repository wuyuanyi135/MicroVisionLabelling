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
from skimage.measure import find_contours, approximate_polygon, subdivide_polygon
def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

img_list = []
an_list = []


dataset_dir =  '/home/wuyuanyi/Desktop/Dataset/Dataset'
batches = [name for name in os.listdir(dataset_dir)
    if os.path.isdir(os.path.join(dataset_dir, name))]
batches_full_path = [os.path.join(dataset_dir, x) for x in batches]

for batch_path in batches_full_path:
    annotation_file_path = os.path.join(batch_path, "via_region_data.csv")
    if not os.path.exists(annotation_file_path):
        # this batch has not been labelled
        continue

    with open(annotation_file_path, "r") as annotation_file:
        annotation = csv.reader(annotation_file)
        next(annotation) # skip the first comment line
        grouped_annotation = itertools.groupby(annotation, lambda x: x[0])
        for image_file, g in grouped_annotation:
            shape_entries = [json.loads(x[5]) for x in g]
            # skip empty files (without label)
            if not bool(shape_entries[0]):
                continue

            # add image
            image_full_path = os.path.join(batch_path, image_file)
            image = skimage.io.imread(image_full_path)
            height, width = image.shape[:2]
            
            

            for i, p in enumerate(shape_entries):
                #mask = np.zeros([height, width],
                #        dtype=np.uint8)
                # Get indexes of pixels inside the polygon and set them to 1
                if p['name'] == 'polygon':
                    rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
                #    mask[rr, cc] = 1

                if p['name'] == 'circle':
                    rr, cc = skimage.draw.circle(p['cy'], p['cx'], p['r'], (height, width))
                    polycoors =  np.stack((rr, cc), axis=1)
                    app_polygon = approximate_polygon(polycoors, 0.1)
                    rr = app_polygon[:,0]
                    cc = app_polygon[:,1]
                #    mask[rr, cc] = 1
                
                xmin = cc.min()
                ymin = rr.min()
                xmax = cc.max()
                ymax = rr.max()

                bbox = (xmin, ymin, xmax-xmin, ymax-ymin)

                # connect to first
                if (rr[0] != rr[-1] or cc[0] != cc[-1]):
                    rr = np.append(rr, rr[0])
                    cc = np.append(cc, cc[0])
                ann = {
                    "iscrowd": False,
                    "area": PolyArea(cc,rr),
                    "bbox": bbox,
                    "category_id": 1,
                    "id": image_file + str(i),
                    "image_id": image_file,
                    "height": height,
                    "width": width,
                    "segmentation": [np.reshape(np.stack([cc, rr], axis=1), (-1)).tolist()]
                }
                an_list.append(ann)
            img_list.append({
                "coco_url": "",
                "date_captured": str(datetime.datetime.now),
                "file_name": image_full_path,
                "flickr_url": "",
                "height": height,
                "id": image_file,
                "license": 1,
                "width": width
            })


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
CATEGORIES = [
    {
        'id': 1,
        'name': 'gb',
        'supercategory': 'object',
    }
]

coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": img_list,
        "annotations": an_list
    }

with open('instances_train2014.json', 'w') as outfile:
    json.dump(coco_output, outfile)
            

