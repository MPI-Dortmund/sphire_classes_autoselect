"""
This file should contain all input/ouput hdf stuff
"""
import h5py
import numpy as np
import os

def write_labeled_hdf(results,output_path,filename):
    """
    The code will generate 2 files with 'output_path' name and suffix '_good' where it'll storage the good results
                            and '_bad' to storage the bad results
    :param results: List of results tuples. The tubles have the format (HDF_PATH,index_in_hdf,label,confidence)
    :param output_path: Path where the results should be written.
    :return: None
    """
    def write_hdf(original_path, out_path, l):
        """
        It copies the image from the original file to the result files.
        :param original_path: hdf starting file
        :param out_path: hdf results file
        :param l: list of index of the images to copy
        :return: None
        """
        counter = 0
        with h5py.File(original_path, driver='core') as original:
            with h5py.File(out_path, 'w') as f:
                group = f.create_group("MDF/images/")

                group.attrs["imageid_max"] = np.array([len(l)-1], dtype=np.int32)

                for original_index in l:
                    subgroup = f.create_group("MDF/images/" + str(counter ) + "/")
                    subgroup.create_dataset("image", data=original['MDF']['images'][str(original_index)]['image'][()])
                    for k, v in original['MDF']['images'][str(original_index )].attrs.items():
                        subgroup .attrs[k]=v
                    counter +=1

    # the given results belong at one starting file
    path_original = results[0][0]
    good_index = [result[1] for result in results if result[2] == 1]
    bad_index = [result[1] for result in results if result[2] == 0]


    write_hdf(path_original, os.path.join(output_path, filename+"_good.hdf"), good_index)
    write_hdf(path_original, os.path.join(output_path, filename + "_bad.hdf"), bad_index)
