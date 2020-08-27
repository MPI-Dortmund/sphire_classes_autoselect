#! /usr/bin/env python
"""
Automatic 2D class selection tool.

MIT License

Copyright (c) 2019 Max Planck Institute of Molecular Physiology

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import multiprocessing
import argparse
import os
import json


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
                "Your configuration file seems to be corrupted. Please check if it is valid."
            )

    input_size = config["model"]["input_size"]
    batch_size = config["train"]["batch_size"]
    good_path = config["train"]["good_path"]
    bad_path = config["train"]["bad_path"]
    pretrained_weights = config["train"]["pretrained_weights"]
    output_file = config["train"]["saved_weights_name"]
    nb_epoch = config["train"]["nb_epoch"]
    nb_epoch_early_stop = config["train"]["nb_early_stop"]
    learning_rate = config["train"]["learning_rate"]
    mask_radius=None
    if "mask_radius" in config["model"]:
        mask_radius = config["model"]["mask_radius"]

    valid_good_path = None
    valid_bad_path = None
    if (
        "valid" in config
        and "good_path" in config["valid"]
        and "bad_path" in config["valid"]
        and config["valid"]["good_path"]
        and config["valid"]["bad_path"]
    ):
        valid_good_path = config["valid"]["good_path"]
        valid_bad_path = config["valid"]["bad_path"]

    if "train_valid_split" in config["train"]:
        train_valid_thresh = config["train"]["train_valid_split"]
    else:
        train_valid_thresh = 0.8

    max_valid_img_per_file = -1
    if "max_valid_img_per_file" in config["train"]:
        if config["train"]["max_valid_img_per_file"] is not None:
            max_valid_img_per_file = config["train"]["max_valid_img_per_file"]

    if input_size[0] % 32 > 0 or input_size[1] % 32 > 0:
        input_size[0] = int(input_size[0] / 32) * 32
        input_size[1] = int(input_size[1] / 32) * 32
        print("Input size has to be a multiple of 32. Changed it to:", input_size)
    from .auto_2d_select import Auto2DSelectNet
    if mask_radius is None:
        mask_radius = input_size[0] * 0.4



    full_rotation_aug = mask_radius != -1
    print("Mask radius is", mask_radius)
    auto2dnet = Auto2DSelectNet(batch_size, input_size, depth=1,mask_radius=mask_radius)

    auto2dnet.train(
        train_good_path=good_path,
        train_bad_path=bad_path,
        save_weights_name=output_file,
        pretrained_weights=pretrained_weights,
        nb_epoch=nb_epoch,
        nb_epoch_early=nb_epoch_early_stop,
        learning_rate=learning_rate,
        train_val_thresh=train_valid_thresh,
        max_valid_img_per_file=max_valid_img_per_file,
        valid_good_path=valid_good_path,
        valid_bad_path=valid_bad_path,
        full_rotation_aug=full_rotation_aug
    )

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    _main_()
