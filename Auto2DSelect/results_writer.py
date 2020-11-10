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

def write_results_to_disk(results, output_path):
    """
    The code will generate 2 files with 'output_path' name and suffix '_good' where it'll storage the good results
                            and '_bad' to storage the bad results
    :param results: List of results tuples. The tubles have the format (image_path,index_in_hdf_or_mrc,label,confidence)
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
        with h5py.File(original_path, 'r', driver="core") as original:
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
    def write_line(out_path, line):
        f = open(out_path, "a+")
        f.write(line+"\n")


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
        bad_confidence = []
        good_confidence = []
        path_original = key
        for _, _ in enumerate(grp):

            if results[running_index][2]==1:
                good_index.append(results[running_index][1])
                good_confidence.append(results[running_index][3])
            else:
                bad_index.append(results[running_index][1])
                bad_confidence.append(results[running_index][3])

            running_index = running_index+1


        '''
        In case of classes, write new classes
        '''
        filename_ext = os.path.basename(path_original).split(".")[-1]
        filename = '.'.join(os.path.basename(path_original).split(".")[:-1])
        if filename_ext == "hdf":
            write_hdf(
                path_original, os.path.join(output_path, filename + "_good.hdf"), good_index
            )

            write_hdf(
                path_original, os.path.join(output_path, filename + "_bad.hdf"), bad_index
            )

        elif filename_ext == "mrcs":

            write_mrcs(
                path_original,
                os.path.join(output_path, filename + "_good.mrcs"),
                good_index,
            )
            write_mrcs(
                path_original, os.path.join(output_path, filename + "_bad.mrcs"), bad_index
            )

        '''
        In case of classes, write a index_confidence file
        '''
        if filename_ext  == "mrcs" or filename_ext  == "hdf":
            for k,ingood in enumerate(good_index):
                write_line(os.path.join(output_path, filename+"_index_confidence.txt"), str(ingood) + " " + "{0:.3f}".format(good_confidence[k]))
            for k, inbad in enumerate(bad_index):
                write_line(os.path.join(output_path, filename+"_index_confidence.txt"),
                           str(inbad) + " " + "{0:.3f}".format(1-bad_confidence[k]))

        '''
        In case of micrographs, write a good.txt and bad.txt with respective filenames.
        Moreover write a filename_confidence file
        '''
        if os.path.basename(path_original).split(".")[-1] == "mrc" or os.path.basename(path_original).split(".")[-1] == "tiff":
            conf = 0
            if results[running_index - 1][2]==1:
                write_line(os.path.join(output_path,"good.txt"),results[running_index-1][0])
                conf = results[running_index-1][3]
            else:
                write_line(os.path.join(output_path, "bad.txt"), results[running_index - 1][0])
                conf = 1-results[running_index - 1][3]

            write_line(os.path.join(output_path, "filename_confidence.txt"), results[running_index - 1][0] + " " + "{0:.3f}".format(conf))

