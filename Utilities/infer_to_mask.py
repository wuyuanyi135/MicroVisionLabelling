import argparse
import requests
import pickle
import os
import glob

parser = argparse.ArgumentParser()
parser.add_argument("server_address")
parser.add_argument("image_path", help="path to the images folder")
parser.add_argument("--ext", help="image extension", default="png")
parser.add_argument("--output-file", help="pickle output", default="infer_result")

args = parser.parse_args()

server_address = args.server_address
ext = args.ext
image_path = args.image_path
output_file = args.output_file


files = glob.glob(os.path.join(image_path, '*.' + ext))
print("Will process {} file(s) using the server {}".format(len(files), server_address))

print("Output file: {}".format(output_file))
out = open(output_file, 'wb')
for i, file in enumerate(files):
    print("Processing: {}; Progress: {}/{}".format(file, i, len(files)))
    with open(file, 'rb') as f:
        ret = requests.post(server_address, f,headers={'Content-Type': 'application/octet-stream'})

        pickle.dump({"f":os.path.basename(file), "m":ret.content}, out)
