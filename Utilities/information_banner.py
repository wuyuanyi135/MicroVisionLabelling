import argparse
import cv2
import os
import glob

parser = argparse.ArgumentParser()

parser.add_argument("mode", help="can be restore or remove")
parser.add_argument("-e", "--extension", help="file extension for the image files", default="jpg")
args = parser.parse_args()

mode = args.mode
assert mode in ["restore", "remove"]

ext = args.extension

dataset_dir =  os.path.join(os.path.dirname(__file__), '../Dataset/')
dataset_dir = os.path.abspath(dataset_dir) # python2 __file__ is not absolute
batches = [name for name in os.listdir(dataset_dir)
    if os.path.isdir(os.path.join(dataset_dir, name))]
batches_full_path = [os.path.join(dataset_dir, x) for x in batches]


for batch_path in batches_full_path:
    if mode == "restore":
        restore_from = os.path.join(batch_path, '.origimgs')
        if not os.path.exists (restore_from):
            print ("Unable to restore images in {}; Use git to go back.".format(restore_from))
            continue
        else:
            for each_file in glob.glob(os.path.join(restore_from, "*."+ext)):
                dest = os.path.join(batch_path, os.path.basename(each_file))
                os.rename(each_file, dest)
            os.removedirs(restore_from)

    if mode == "remove":
        save_to = os.path.join(batch_path, '.origimgs')
        if os.path.exists(save_to):
            print("{} contains .origimgs. It may have been processed. Skip".format(save_to))
            continue
        else:
            os.mkdir(save_to)
            for each_file in glob.glob(os.path.join(batch_path, "*."+ext)):
                img = cv2.imread(each_file)
                img = img[0:1024, :]
                dest = os.path.join(save_to, os.path.basename(each_file))
                os.rename(each_file, dest)
                cv2.imwrite(each_file, img)