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
import os
import h5py
import numpy as np
import mrcfile

# pylint: disable=C0330, C0301

def write_labeled_hdf(results, output_path, filename):
    """
    The code will generate 2 files with 'output_path' name and suffix '_good' where it'll storage the good results
                            and '_bad' to storage the bad results
    :param results: List of results tuples. The tubles have the format (image_path,index_in_hdf,label,confidence)
    :param output_path: Path where the results should be written.
    :return: None
    """

    def write_hdf(original_path, out_path, image_indices):
        """
        It copies the image from the original file to the result files.
        :param original_path: hdf starting file
        :param out_path: hdf results file
        :param image_indices: list of index of the images to copy
        :return: None
        """
        counter = 0
        with h5py.File(original_path, driver="core") as original:
            with h5py.File(out_path, "w") as f:
                group = f.create_group("MDF/images/")

                group.attrs["imageid_max"] = np.array([len(image_indices) - 1], dtype=np.int32)

                for original_index in image_indices:
                    subgroup = f.create_group("MDF/images/" + str(counter) + "/")
                    subgroup.create_dataset(
                        "image",
                        data=original["MDF"]["images"][str(original_index)]["image"][
                            ()
                        ],
                    )
                    for k, v in original["MDF"]["images"][
                        str(original_index)
                    ].attrs.items():
                        subgroup.attrs[k] = v
                    counter += 1

    def write_mrcs(original_path, out_path, l):
        """

        :param original_path:
        :param out_path:
        :param l:
        :return:
        """
        if l:
            with mrcfile.mmap(original_path, permissive=True, mode="r") as original_mrc:
                with mrcfile.new(out_path) as new_mrc:
                    new_mrc.set_data(original_mrc.data[l])
    def write_line(out_path, results):
        f = open(out_path, "a+")
        f.write(results[0]+"\n")


    # the given results belong at one starting file
    # Sort results
    results = sorted(results, key=lambda x: x[0])

    # Group them by filename
    image_paths = [result[0] for result in results]
    import itertools
    running_index = 0
    for index, (key, group) in enumerate(itertools.groupby(image_paths)):
        grp = list(group)
        good_index = []
        bad_index = []
        path_original = key
        for _, _ in enumerate(grp):

            if results[running_index][2]==1:
                good_index.append(results[running_index][1])
            else:
                bad_index.append(results[running_index][1])

            running_index = running_index+1

        if os.path.basename(path_original).split(".")[1] == "hdf":
            write_hdf(
                path_original, os.path.join(output_path, filename + "_good.hdf"), good_index
            )
            write_hdf(
                path_original, os.path.join(output_path, filename + "_bad.hdf"), bad_index
            )
        elif os.path.basename(path_original).split(".")[1] == "mrcs":
            write_mrcs(
                path_original,
                os.path.join(output_path, filename + "_good.mrcs"),
                good_index,
            )
            write_mrcs(
                path_original, os.path.join(output_path, filename + "_bad.mrcs"), bad_index
            )

        # Write indices
        if results[running_index - 1][2]==1:
            write_line(os.path.join(output_path,"good.txt"),results[running_index-1])
        else:
            write_line(os.path.join(output_path, "bad.txt"), results[running_index - 1])

