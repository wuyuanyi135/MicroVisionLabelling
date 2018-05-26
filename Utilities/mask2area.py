import pickle
import json
import skimage.measure as measure
from pycocotools import coco, mask

output_file = open('area.p', 'wb')

input_file = open('infer_result', 'rb')

try:
    while True:
        obj = pickle.load(input_file)

        filename = obj["f"]
        try:
            timestamp = int(filename.split('.')[0])
        except Exception:
            continue
            
        m = json.loads(obj["m"])
        
        segms = m['cls_segms']
        if segms is None:
            continue
        alpha = mask.area(segms[1])
        beta = mask.area(segms[2])
        pickle.dump({"timestamp": timestamp, "alpha": alpha, "beta": beta}, output_file, protocol=2)
except EOFError:
    print("File read end")