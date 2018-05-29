import json
import pickle
from matplotlib import pyplot as plt
import numpy as np
from pycocotools import mask as cocoMask
import itertools
import cv2
import scipy.io
import time

from multiprocessing import Process, Pool, BoundedSemaphore
import multiprocessing

def mask2ellipse(m):
    ret = []
    if not len(m):
        return ret
    masks = cocoMask.decode(m)
    for j in range(masks.shape[-1]):
        mask = masks[:,:,j]
        _, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        #print("Contour found {}".format(len(contours)))
        if not len(contours):
            continue
        contours = sorted(contours, reverse=True, key=cv2.contourArea)
        #print("Contour sorted")
        # only process the greatest contour
        contour = contours[0]
        if len(contour) < 5:
            continue
        e = cv2.fitEllipse(contour)
        #print("Ellipse found")
        ret.append((e[1], cv2.contourArea(contour)))
    return ret

window_size = 120
ellipse_list = []

def process_task(lock, t, load_list, counter):
    #only process one time point
    alpha_list = []
    beta_list = []
    print("Processing {} ({} objs)".format(counter, len(load_list)))
    start = time.time()
    for loaded in load_list:
        
        
        alpha_list.extend(mask2ellipse(loaded[1]))
        beta_list.extend(mask2ellipse(loaded[2]))
        
    end = time.time()
    print ("Finished {} ({} objs) takes {}".format(counter, len(load_list), end-start))

    lock.release()
    return (t,alpha_list, beta_list)
if __name__=="__main__":
    try:
        with open('infer_result_remove_overlapping' ,'rb') as input_file:
            manager = multiprocessing.Manager()

            
            counter = 0
            pool = Pool() 
            lock = manager.BoundedSemaphore(4)
            while True:
                try:
                    t_list = []
                    load_list = []
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
                    print ("Prepared an object: {} containing {} things in the list".format(counter, len(load_list)))
                    
                    
                    lock.acquire()
                    result = pool.apply_async(process_task, (lock, t, list(load_list), counter), callback=ellipse_list.append)
                    counter += 1
                    # result.get()
                except EOFError as e:
                    break
                    
            pool.close()
            pool.join()
        print("Writing matlab file...")
        scipy.io.savemat("ellipses", {"ellipse": [x[1] for x in ellipse_list], 'time': [x[0] for x in ellipse_list]})
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()