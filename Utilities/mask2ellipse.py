import pickle
import json
import cv2
from pycocotools import coco, mask

output_file = open('ellipse.p', 'wb')

input_file = open('infer_result', 'rb')
def seg2ellipse(segs):
    ellipses = []
    for each_seg in [mask.decode(seg) for seg in segs]:
        _, c,h = cv2.findContours(each_seg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in c:
            if len(contour) < 5:
                print("Malformed contour detected (length of contour = {}), skip".format(len(contour)))
                continue
            ellipse = cv2.fitEllipse(contour)
            ellipses.append(ellipse)
    return ellipses
    
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
        # alpha 
        alpha_ellipses = seg2ellipse(segms[1])
        beta_ellipses = seg2ellipse(segms[2])
        pickle.dump({"timestamp": timestamp, "alpha_ellipses": alpha_ellipses, "beta_ellipses": beta_ellipses}, output_file, protocol=2)
except EOFError:
    print("File read end")