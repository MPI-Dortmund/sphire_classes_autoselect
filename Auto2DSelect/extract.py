
import multiprocessing
import argparse
from Auto2DSelect.helper import get_key_list_images, getImages_fromList_key
from Auto2DSelect.auto_2d_select import  get_relevant_slices
import numpy as np
import mrcfile

ARGPARSER = argparse.ArgumentParser(
    description="Extracts central slices of each tomogram from .hdf tomogram stack",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

ARGPARSER.add_argument("-i", "--input", required=True, help="Path to input hdf file.")
ARGPARSER.add_argument("-o", "--output", required=True, help="Path to output mrcs file.")
ARGPARSER.add_argument(
        "--invertimg",
        action="store_true",
        help="Invert tall he images"
    )

def _main_():
    args = ARGPARSER.parse_args()

    input_file = args.input
    output_file = args.output
    doinvert = True
    print("Read images")
    key_img_list = get_key_list_images(input_file)
    data_tubles = [(input_file,i) for i in key_img_list]
    images = getImages_fromList_key(data_tubles)
    print("Extract slices")
    images = [get_relevant_slices(img) for img in images]
    if doinvert:
        print("Invert")
        images = [invert(img) for img in images]
    print("Write")
    images = np.array(images)

    with mrcfile.new(output_file) as new_mrc:
        new_mrc.set_data(images)

def invert(img):
    img = np.max(img)-img
    return img

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    _main_()