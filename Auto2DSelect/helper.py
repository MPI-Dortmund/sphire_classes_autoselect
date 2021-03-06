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

from os import path, listdir
import h5py
from PIL import Image  # install it via pip install pillow
import numpy as np
import mrcfile

"""
The format of the .hf file is the following:
    ['MDF']['images']['i']['image']   where i is a number representing the i-th images
hence to get the images number 5:
    ['MDF']['images']['5']['image'][()]
"""


def create_circular_mask(h, w, center=None, radius=None):

    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask

def checkfiles(path_to_files):
    """
    checks if the hdf files are in the correct path and returns True if all of them exists
    :param path_to_files:    list of paths
    :return:
    """
    if isinstance(path_to_files, (list, tuple)):
        for p in path_to_files:
            if not path.isfile(p):
                return False
    elif isinstance(path_to_files, str):
        return path.isfile(path_to_files)
    return True


def calc_2d_spectra(img):
    from scipy import fftpack
    import numpy as np

    F1 = fftpack.fft2(img)
    F2 = fftpack.fftshift(F1)
    psd2D = np.abs(F2) ** 2

    return psd2D


def getList_files(paths):
    """
    Returns the list of the valid hdf files in the given paths. It is called recursively
    :param paths: path or list of paths
    :return:
    """
    if isinstance(paths, str):
        paths = [paths]
    list_new_paths = list()
    iterate = False
    for p in paths:
        if path.isdir(p):
            iterate = True
            list_new_paths += [path.join(p, f) for f in listdir(p)]
        elif path.isfile(p):
            list_new_paths.append(p)
        else:
            print(
                "WARNING: The given path '"
                + str(p)
                + "' is not a folder or a file and it will be ignored"
            )
    if iterate is True:
        return getList_files(list_new_paths)
    return list_new_paths


def getList_relevant_files(path_to_files):
    """
    Check if the given files are hdf/mrcs/st with a valid format. Return The list of valid hdf
    :param path_to_files: list of all the files present in the folder (and subfolder)given from the user
    :return: list of valid hdf
    """

    return [
        path_to_file
        for path_to_file in path_to_files
        if path_to_file.endswith("mrcs")
        or path_to_file.endswith("mrc")
        or path_to_file.endswith("st")
        or h5py.is_hdf5(path_to_file)
    ]


""" FUNCTION TO READ THE HDF"""

def get_key_list_images(path):
    """
    Returns the list of the keys representing the images in the hdf/mrcs/st file. It will be converted in list of integer
    :param path:
    :return:
    """
    print("Try to list images on", path)
    import os
    filename_ext = os.path.basename(path).split(".")[-1]
    result_list = None
    try:
        if filename_ext in {"mrcs", "st"}:
            with mrcfile.mmap(path, permissive=True, mode="r") as mrc:
                list_candidate = [i for i in range(mrc.header.nz)]
                if len(list_candidate) > 0:
                    result_list = list_candidate
        if filename_ext == "mrc":
            with mrcfile.mmap(path, permissive=True, mode="r") as mrc:
                result_list = list(range(1))
    except Exception as e:
        print(e)
        print(
            "WARNING in get_list_images: the file '"
            + path
            + " is not an valid mrc file. It will be ignored"
        )

    if filename_ext == "hdf":
        try:
            with h5py.File(path, "r") as f:
                list_candidate = [int(v) for v in list(f["MDF"]["images"])]
        except:
            print(
                "WARNING in get_list_images: the file '"
                + path
                + " is not an HDF file with the following format:\n\t['MDF']['images']. It will be ignored"
            )
        if len(list_candidate) > 0:
            result_list = list_candidate
    return result_list


def getImages_fromList_key(file_index_tubles):
    """
    Returns the images in the hdf file (path_to_file) listed in (list_images)
    :param path_to_file:    path to hdf file
    :param list_images: list of keys of the DB. It is the output( or part of its) given from 'get_list_images'
    :return: Returns a list of numpy arrays
    """
# driver="core"
    result_data = list()
    for path_to_file, list_images in file_index_tubles:
        data = list()
        if path.isfile(path_to_file):
            if path.basename(path_to_file).split(".")[-1] == "hdf":
                try:
                    with h5py.File(path_to_file, 'r') as f:
                        if isinstance(list_images, list) or isinstance(
                            list_images, tuple
                        ):
                            data = [
                                np.nan_to_num(f["MDF"]["images"][str(i)]["image"][()])
                                for i in list_images
                            ]  # [()] is used instead of .value
                        elif isinstance(list_images, int):
                            data = np.nan_to_num(f["MDF"]["images"][str(list_images)]["image"][()])
                        else:
                            print(
                                "\nERROR in getImages_fromList_key: invalid list_images, it should be a string or a list/tuple of strings:",
                                type(list_images),
                            )
                            print("you try to get the following images")
                            print(list_images)
                            exit()
                except Exception as e:
                    print(e)
                    print(
                        "\nERROR in getImages_fromList_key: the file '"
                        + path_to_file
                        + " is not an HDF file with the following format:\n\t['MDF']['images']['0']['image']"
                    )
                    print("you try to get the following images")
                    print(list_images)
                    print("there are " + str(len(f["MDF"]["images"])))
                    exit()
            elif path.basename(path_to_file).split(".")[-1] in ["mrc", "mrcs", "st"]:
                data = []
                with mrcfile.mmap(path_to_file, permissive=True, mode="r") as mrc:

                    if isinstance(list_images, int):
                        list_images = [list_images]

                    if isinstance(list_images, list) or isinstance(list_images, tuple):
                        if mrc.header.nz > 1:
                            if len(list_images)==1:
                                data = np.nan_to_num(mrc.data[list_images[0]])
                            else:
                                data = [np.nan_to_num(mrc.data[i]) for i in list_images]
                        elif len(list_images) == 1:
                            data = np.nan_to_num(mrc.data)

        result_data.append(data)



    return result_data


def getImages_fromList_key_old(path_to_file, list_images):
    """
    Returns the images in the hdf file (path_to_file) listed in (list_images)
    :param path_to_file:    path to hdf file
    :param list_images: list of keys of the DB. It is the output( or part of its) given from 'get_list_images'
    :return: Returns a list of numpy arrays
    """
    data = list()
    if path.isfile(path_to_file):
        if path.basename(path_to_file).split(".")[-1] == "hdf":
            try:
                with h5py.File(path_to_file, driver="core") as f:
                    if isinstance(list_images, list) or isinstance(list_images, tuple):
                        data = [
                            f["MDF"]["images"][str(i)]["image"][()] for i in list_images
                        ]  # [()] is used instead of .value
                    elif isinstance(list_images, int):
                        data = f["MDF"]["images"][str(list_images)]["image"][()]
                    else:
                        print(
                            "\nERROR in getImages_fromList_key: invalid list_images, it should be a string or a list/tuple of strings:",
                            type(list_images),
                        )
                        print("you try to get the following images")
                        print(list_images)
                        exit()
            except Exception as e:
                print(e)
                print(
                    "\nERROR in getImages_fromList_key: the file '"
                    + path_to_file
                    + " is not an HDF file with the following format:\n\t['MDF']['images']['0']['image']"
                )
                print("you try to get the following images")
                print(list_images)
                print("there are " + str(len(f["MDF"]["images"])))
                exit()
        elif path.basename(path_to_file).split(".")[-1] in ["mrc", "mrcs", "st"]:
            data = []
            with mrcfile.mmap(path_to_file, permissive=True, mode="r") as mrc:

                if isinstance(list_images, int):
                    list_images = [list_images]

                if isinstance(list_images, list) or isinstance(list_images, tuple):
                    if mrc.header.nz > 1:
                        data = [mrc.data[i] for i in list_images]
                    elif len(list_images) == 1:
                        data = [mrc.data]
    return data


""" FUNCTION TO MANIPULATE THE IMAGES"""


def apply_mask(img, mask):
    mean = np.mean(img)
    img[mask==False]=mean
    return img


def resize_img(img, resize=(76, 76)):
    """
    Resize the given image into the given size
    :param img: as numpy array
    :param resize: resize size
    :return: return the resized img
    """
    im = Image.fromarray(img)
    return np.array(im.resize(resize, resample=Image.BILINEAR))


def normalize_img(img):
    """
    normalize the images in base of its mean and variance
    :param img:
    :return:
    """
    import numpy as np

    # img = img.astype(np.float64, copy=False)
    mean = np.mean(img)
    std = np.std(img)
    img = (img - mean) / (std+0.00001)
    # img = img.astype(np.float32, copy=False)
    return img


def flip_img(img, t=None):
    """
    It flip the image in function of the given typ
    :param img:
    :param t: type of the flip
                1 --> flip over the row. Flipped array in up-down direction.(X)
                2 --> flip over the column Flipped array in right-left direction(Y)
                3 --> flip over the column and the row (X and Y)
                otherwise --> no flip
    :return:
    """
    if t == 1:
        return np.flipud(img)
    elif t == 2:
        return np.fliplr(img)
    elif t == 3:
        return np.flipud(np.fliplr(img))
    return img
