'''
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

'''
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
