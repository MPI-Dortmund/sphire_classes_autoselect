#! /usr/bin/env python
import json
import multiprocessing
import argparse
from .auto_2d_select import Auto2DSelectNet
from . import hdf_io
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
try:
    os.environ["CUDA_VISIBLE_DEVICES"]
except KeyError:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

ARGPARSER = argparse.ArgumentParser(
    description="Train auto 2D class selection",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)


ARGPARSER.add_argument("-i", "--input", required=True, help="Path folder to input HDF.")

ARGPARSER.add_argument("-o", "--output", required=True, help="Path to output folder.")

ARGPARSER.add_argument("-w", "--weights", required=True, help="Path network weights.")

ARGPARSER.add_argument("-t", "--confidence_threshold", default=0.5, type=float, help="Classes with a confidence higher as that threshold are classified as good.")

ARGPARSER.add_argument("--gpu", default=-1, type=int, help="GPU to run on.")

ARGPARSER.add_argument(
    "-c", "--config", required=True, help="Path to config file."
)

def _main_():
    args = ARGPARSER.parse_args()
    input_path = args.input
    weights_path = args.weights
    output_path = args.output
    threshold = args.confidence_threshold

    if args.gpu != -1:
        str_gpu = str(args.gpu)
        os.environ["CUDA_VISIBLE_DEVICES"] = str_gpu

    with open(args.config) as config_buffer:
        try:
            config = json.load(config_buffer)
        except json.JSONDecodeError:
            print("Your configuration file seems to be corruped. Please check if it is valid.")


    input_size = config["model"]["input_size"]
    batch_size = config["train"]["batch_size"]

    auto2dnet = Auto2DSelectNet(batch_size, input_size)
    result = auto2dnet.predict(input_path, weights_path, good_thresh=threshold)

    hdf_io.write_labeled_hdf(result, output_path,os.path.basename(input_path).split('.')[0])
    good = []
    bad = []
    for res in result:
        if res[2]:
            good.append(res[1])
        else:
            bad.append(res[1])
    good.sort()
    bad.sort()
    print("\n Good classes (",len(good),")", good, "\n")
    print("\n Bad classes (",len(bad),")", bad,"\n")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    _main_()
