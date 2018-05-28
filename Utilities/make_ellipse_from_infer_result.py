import json
import pickle
from matplotlib import pyplot as plt
import numpy as np
from pycocotools import mask as cocoMask
import itertools
import cv2
import scipy.io
from multiprocessing import Process, Pool, BoundedSemaphore
import multiprocessing

def mask2ellipse(m):
    ret = []
    if not len(m):
        return ret
    masks = cocoMask.decode(m)
    for mask in masks:
        _, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not len(contours):
            continue
        contours = sorted(contours, reverse=True, key=cv2.contourArea)
        # only process the greatest contour
        contour = contours[0]
        if len(contour) < 5:
            continue
        e = cv2.fitEllipse(contour)
        ret.append((e[1], cv2.contourArea(contour)))
    return ret

window_size = 120
ellipse_list = []

def process_task(lock, t, load_list, counter):
    #only process one time point
    alpha_list = []
    beta_list = []
    print("Processing {}".format(counter))
    for loaded in load_list:
        alpha_list.extend(mask2ellipse(loaded[1]))
        beta_list.extend(mask2ellipse(loaded[2]))
    lock.release()
    print("Processed {}".format(counter))
    return (t,alpha_list, beta_list)
if __name__=="__main__":
    with open('infer_result_remove_overlapping' ,'rb') as input_file:
        load_list = []
        t_list = []
        counter = 0
        pool = Pool(processes=3) 
        lock = multiprocessing.Manager().BoundedSemaphore(3)
        while True:
            try:
                for i in range(window_size):
                    obj = pickle.load(input_file)
                    m = json.loads(obj['m'])['cls_segms']
                    try:
                        f = int(obj['f'].split('.')[0])
                    except:
                        continue
                    t_list.append(f)
                    if not m:
                        continue
                    load_list.append(m)
                t = np.mean(t_list)
                print ("Prepared an object: {}".format(counter))
                counter += 1
                
                lock.acquire()
                
                result = pool.apply_async(process_task, (lock, t, load_list, counter), callback=ellipse_list.append)
            except Exception as e:
                raise(e)
        pool.close()
        pool.join()
    print("Writing matlab file...")
    scipy.io.savemat("ellipses", {"ellipse": ellipse_list})