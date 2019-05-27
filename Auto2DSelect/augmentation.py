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

import random
import numpy as np
from scipy import ndimage
from . import helper


class Augmentation:
    """
    Class for doing data augmentation
    """

    def __init__(self, is_grey=False):
        self.is_grey = is_grey

    def image_augmentation(self, image):
        """
        Applies random selection of data augmentations
        :param image:  Input image
        :return: Augmented image
        """
        augmentations = [
            self.additive_gauss_noise,
            self.add,
            self.contrast_normalization,
            self.multiply,
            # self.sharpen,
            self.dropout,
        ]

        num_augs = np.random.randint(0, np.minimum(6, len(augmentations)))
        if num_augs > 0:

            if np.random.rand() > 0.5:
                augmentations.append(self.gauss_blur)
            else:
                augmentations.append(self.avg_blur)

            selected_augs = random.sample(augmentations, num_augs)
            image = image.astype(np.float32, copy=False)
            for sel_aug in selected_augs:
                image = sel_aug(image)
            #   print "Mean after", sel_aug, " sum: ", np.mean(image)
            if self.is_grey:
                min_img = np.min(image)
                max_img = np.max(image)
                image = ((image - min_img) / (max_img - min_img)) * 255
                #    image = np.clip(image, 0, 255)
                image = image.astype(np.uint8, copy=False)

        return helper.flip_img(image, random.randint(0, 3))

    def gauss_blur(self, image, sigma_range=(0, 3)):
        """
        Applys gaussian blurring with random sigma
        :param image: Input image
        :param sigma_range:  Range for random sigma
        :return: Blurred image
        """
        rand_sigma = sigma_range[0] + np.random.rand() * (
            sigma_range[1] - sigma_range[0]
        )
        result = ndimage.gaussian_filter(image, sigma=rand_sigma, mode="nearest")

        if not np.issubdtype(image.dtype, np.float32):
            result = result.astype(np.float32, copy=False)
        return result

    def avg_blur(self, image, kernel_size=(2, 7)):
        """
        Applys average blurring with random kernel size
        :param image: Input image (numpy array)
        :param kernel_size: Range for random kernel size
        :return: Blurred image
        """
        rang_kernel_size = np.random.randint(kernel_size[0], kernel_size[1])
        image = ndimage.uniform_filter(image, size=rang_kernel_size, mode="nearest")
        return image

    def additive_gauss_noise(self, image, max_sigma_range_factor=0.05):
        """
        Add random gaussian noise to image
        :param image: Input image
        :param max_sigma_range_factor: Range for max_sigma_range. The standard deviation of the noise is
        choosen randomly depending on the standard deviation of the image.
        The choosen standard deviation for noise is between: 0 and  max_sigma_factor*6*np.std(image)
        :return:
        """

        width = 2 * 3 * np.std(image)
        max_sigma = width * max_sigma_range_factor
        rand_sigma = np.random.rand() * max_sigma

        noise = np.random.randn(image.shape[0], image.shape[1])
        np.multiply(noise, rand_sigma, out=noise)
        np.add(image, noise, out=image)

        if not np.issubdtype(image.dtype, np.float32):
            image = image.astype(np.float32, copy=False)

        return image

    def contrast_normalization(self, image, alpha_range=(0.5, 2.0)):
        """
        Spread or squeeze the pixel values.
        :param image: Input image
        :param alpha_range: Range for alpha. Alpha controlls the normalization.
        :return: Modified image
        """
        rand_multiplyer = alpha_range[0] + np.random.rand() * (
            alpha_range[1] - alpha_range[0]
        )
        middle = np.median(image)
        np.subtract(image, middle, out=image)
        np.multiply(rand_multiplyer, image, out=image)
        np.add(middle, image, out=image)

        return image

    def dropout(self, image, ratio=(0.01, 0.1)):
        """
        Set a random selection of pixels to the mean of the image
        :param image: Input image
        :param ratio: Range for random ratio
        :return: Modified image
        """
        if isinstance(ratio, float):
            rand_ratio = ratio
        else:
            rand_ratio = ratio[0] + np.random.rand() * (ratio[1] - ratio[0])
        mean_val = np.mean(image)
        drop = np.random.binomial(
            n=1, p=1 - rand_ratio, size=(image.shape[0], image.shape[1])
        )
        image[drop == 0] = mean_val

        return image

    def add(self, image, scale=0.05):
        """
        Adds a random constant to the image
        :param image: Input image
        :param scale: Scale for random constant. The random constant
        will be between 0 and scale*6*std(image)
        :return: Modified image
        """
        width = 2 * 3 * np.std(image)
        width_rand = scale * width
        rand_constant = (np.random.rand() * width_rand) - width_rand / 2
        np.add(image, rand_constant, out=image)

        return image

    def multiply(self, image, mult_range=(0.5, 1.5)):
        """
        Multipy the input image by a random float.
        :param image: Input image
        :param mult_range: Range for random multiplier
        :return: multiplied image
        """

        rand_multiplyer = mult_range[0] + np.random.rand() * (
            mult_range[1] - mult_range[0]
        )
        np.multiply(image, rand_multiplyer, out=image)
        return image
