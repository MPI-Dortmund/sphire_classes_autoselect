from auto_2d_select import Auto2DSelectNet


#! /usr/bin/env python
import multiprocessing


def _main_():
    good_path = "/home/twagner/Projects/Auto2DSelect/data/2019-05-20/GOOD_CLASSES/2018_12_14_alphaActin_aged_Phalloidin_Sabrina/Good.hdf"
    bad_path = "/home/twagner/Projects/Auto2DSelect/data/2019-05-20/BAD_CLASSES/2018_12_14_alphaActin_aged_Phalloidin_Sabrina/contamination.hdf"
    input_size = (75, 75)
    batch_size = 32
    auto2dnet = Auto2DSelectNet(good_path, bad_path, batch_size, input_size)
    auto2dnet.train()
    pass


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    _main_()
