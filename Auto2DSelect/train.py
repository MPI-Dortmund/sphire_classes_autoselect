#! /usr/bin/env python
import multiprocessing
import argparse
import os
import json
from .auto_2d_select import Auto2DSelectNet


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

ARGPARSER.add_argument("-c", "--config", required=True, help="Path to config file.")

ARGPARSER.add_argument("--gpu", default=-1, type=int, help="GPU to run on.")


def _main_():
    args = ARGPARSER.parse_args()

    if args.gpu != -1:
        str_gpu = str(args.gpu)
        os.environ["CUDA_VISIBLE_DEVICES"] = str_gpu

    with open(args.config) as config_buffer:
        try:
            config = json.load(config_buffer)
        except json.JSONDecodeError:
            print(
                "Your configuration file seems to be corruped. Please check if it is valid."
            )

    input_size = config["model"]["input_size"]
    batch_size = config["train"]["batch_size"]
    good_path = config["train"]["good_classes"]
    bad_path = config["train"]["bad_classes"]
    pretrained_weights = config["train"]["pretrained_weights"]
    output_file = config["train"]["saved_weights_name"]
    auto2dnet = Auto2DSelectNet(batch_size, input_size)

    auto2dnet.train(
        good_path,
        bad_path,
        save_weights_name=output_file,
        pretrained_weights=pretrained_weights,
    )


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    _main_()
