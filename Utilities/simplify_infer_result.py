"""
Use this file to post-process the infer result
1. break up the multi-body masks
2. remove multi body masks
3. remove overlapping objects (by threshold)

Warning: after processing, cls_boxes is no longer working.
"""

import os
import argparse
import cv2
import skimage.measure
import logging
import json
import pickle
import numpy as np
from pycocotools import mask
import time
import itertools
import skimage.draw
handler = logging.StreamHandler()
formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)

logger = logging.Logger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("infer_result", help="path to infer_result", default="./infer_result")
arg_parser.add_argument("output_result", help="path to the output file", default="./infer_result_filtered")
arg_parser.add_argument("-b" , "--break-up", help="break up the multi-part masks", default=True, action="store_true")
arg_parser.add_argument("-t", "--threshold", help="threshold to remove the overlapping objects", default=0.8, type=float)
arg_parser.add_argument("-j", "--multiprocessing", help="use multiprocessing to boost the computation.", default=0, type=int)
#arg_parser.add_argument("-a", "--remove-all", help="remove all multibody objects, otherwise keep only the greatest one in area", default=False, action="store_true")
#arg_parser.add_argument("-r", "--remove-multibody", help="enable filtering out the multibody object. Conflict with --break-up", default=False, action="store_true")

args = arg_parser.parse_args()

infer_result = args.infer_result
output_result = args.output_result
threshold = args.threshold
assert threshold < 1 and threshold > 0

mp = args.multiprocessing
assert mp == False or mp > 0
break_up = args.break_up
def main ():
    
    logger.info("Reading the raw network output file: {}".format(infer_result))
    logger.info("writing the output to: {}".format(output_result))
    with open(infer_result, 'rb') as input_file, open(output_result, 'wb') as output_file:
        while True:
            try:
                obj = pickle.load(input_file)
                f = obj["f"]
                try:
                    m = json.loads(obj["m"])
                except Exception as e:
                    m = obj["m"]
                    print(e)
                segms = m["cls_segms"]
                bbox = m["cls_boxes"]
                if not segms:
                    # let this one go
                    continue
                
                processed, addition, removal = remove_overlapping_cross_classes(segms, break_parts=break_up, threshold=threshold, bbox=bbox)
                m["cls_segms"] = processed # do not stringify, just save the object.
                obj["m"] = m
                logger.info("{}: +{} -{}".format(f, addition, removal))

                pickle.dump(obj, output_file)
            except EOFError:
                break

# process one obj and return the filtered object
def mp_worker(obj, break_up, threshold, lock):
    try:
        start = time.time()
        f = obj["f"]
        try:
            m = json.loads(obj["m"])
        except:
            m = obj["m"]
        segms = m["cls_segms"]
        bbox = m["cls_boxes"]
        if not segms or not len(segms):
            # let this one go
            lock.release()
            return None
        processed, addition, removal = remove_overlapping_cross_classes(segms, break_parts=break_up, threshold=threshold, bbox=bbox)
        m["cls_segms"] = processed # do not stringify, just save the object.
        obj["m"] = m
        lock.release()
        end = time.time()
        return obj, addition, removal, end-start
    except Exception as e:
        lock.release()
        print (e)
        return None

def main_mp():
    import multiprocessing
    multiprocessing.log_to_stderr()
    logger.info("Reading the raw network output file: {}".format(infer_result))
    logger.info("writing the output to: {}".format(output_result))
    logger.info("Executing in multiprocessing mode")
    
    logger.info("creating pool with {} worker(s)".format(mp))
    pool = multiprocessing.Pool(processes=mp)
    manager = multiprocessing.Manager()
    counter = 0
    lock = manager.BoundedSemaphore(value=mp)
    try:
        with open(infer_result, 'rb') as input_file, open(output_result, 'wb') as output_file:
                def cb(values):
                    if not values:
                        logger.info('A task has been skipped due to error')

                        return
                    obj = values[0]
                    segms = obj["m"]["cls_segms"]
                    mask_len = sum([len(x) for x in segms])
                    logger.info("{}: +{} -{}; len(mask)={}; time={}s".format(obj["f"], values[1], values[2], mask_len, values[3]))
                    pickle.dump(obj, output_file)
                while True:
                    try:
                        obj = pickle.load(input_file)


                        lock.acquire()
                        logger.info("Add new task to the queue: {}, counter: {}".format(obj["f"], counter))
                        counter +=1
                        result = pool.apply_async(mp_worker, (obj, break_up, threshold, lock), callback=cb)
                        #result.get()
                    except EOFError:
                        logger.info("Done")
                        break

                logger.info("Clearing the pool")
                pool.close()
                pool.join()
    except (KeyboardInterrupt) as e:
        logger.info("Stopping...")
        logger.error(str(e))
        pool.terminate()
        pool.join()
    
# return the sorted encoded masks
def sort_by_mask_area(masks):
    return sorted(masks, reverse=True, key=mask.area)

def overlapping_percentage(mask1, mask2):
    areas = min(mask.area([mask1, mask2]))
    if areas == 0:
        return 0
    percentage = mask.area(mask.merge([mask1, mask2], intersect=True))/areas
    return percentage 

# only assess the top 'process_limit' number of largest masks.
def remove_overlapping_small_objects(masks, threshold=0.9, process_limit = 1000):
    if len(masks) == 0:
        return []
    deleted_index = []
    sorted_masks = sort_by_mask_area(masks)
    
    for i, current_mask in enumerate(sorted_masks):
        if i in deleted_index:
            continue
        if i > process_limit:
            break
        for j in range(i+1, len(sorted_masks)):
            if j in deleted_index:
                continue
            test_mask = sorted_masks[j]
            overlapping = overlapping_percentage(current_mask, test_mask)
            if overlapping > threshold:
                deleted_index.append(j)
    return [m for i, m in enumerate(sorted_masks) if i not in deleted_index], len(deleted_index)

def break_up_masks(masks):
    ret = []
    if not len(masks):
        return ret
    addition = 0

    decoded_masks = mask.decode(masks)
    for i in range(decoded_masks.shape[-1]):
        decoded_mask = decoded_masks[:, :, i]
        _, c, _ = cv2.findContours(decoded_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(c) == 1:
            ret.append(masks[i])
        else:
            for contour in c:
                canvas = np.zeros(decoded_masks.shape)
                reshaped = contour.reshape(-1,2)
                x = reshaped[:,0]
                y = reshaped[:,1]
                rr,cc = skimage.draw.polygon(y, x)
                canvas[rr, cc] = 1
                new_mask = mask.encode(np.asfortranarray(canvas.astype(np.uint8)))[0]
                try:
                    new_mask['is'] = masks[i]['is']
                    new_mask['p'] = masks[i]['p']
                except Exception as e:
                    logging.error(str(e))
                ret.append(new_mask)
                addition += 1
    return ret, addition
def remove_overlapping_cross_classes(multiclass_masks, threshold = 0.9, process_limit = 1000, break_parts = False, bbox=None):
    
    #assign attribute for classification
    for i, masks_of_one_class in enumerate(multiclass_masks):
        for j, mask in enumerate(masks_of_one_class):
            mask['is'] = i
            if bbox:
                try:
                    mask['p']=bbox[i][j][-1]
                except Exception as e:
                    mask['p']=None
                    logger.error(str(e))
        
    # flatten all masks in one array
    combined_masks = []
    [combined_masks.extend(masks_of_one_class) for masks_of_one_class in multiclass_masks]
    if break_parts:
        combined_masks, addition = break_up_masks(combined_masks)
    processed_masks, removal = remove_overlapping_small_objects(combined_masks, threshold, process_limit)
   
    ret =  [];
    for i in range(len(multiclass_masks)):
        ret.append([mask for mask in processed_masks if mask['is'] == i])
    return ret, addition, removal
if __name__ == '__main__':
    if mp:
        main_mp()
    else:
        main()
    
    