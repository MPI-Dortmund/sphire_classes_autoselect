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
# pylint: disable=C0330, C0301

import os
import multiprocessing
from keras.utils import Sequence
from keras.layers import (
    Conv2D,
    Input,
    MaxPooling2D,
    BatchNormalization,
    Dense,
    UpSampling2D,
    GlobalAveragePooling2D,
    Dropout,
    Flatten
)
from keras.layers.merge import concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard,
)

# from swa.keras import SWA
import numpy as np
from Auto2DSelect.SGDRSchedular import SGDRScheduler
from .helper import (
    getImages_fromList_key,
    resize_img,
    normalize_img,
    apply_mask,
    get_key_list_images,
    getList_relevant_files,
    getList_files,
    calc_2d_spectra
)
from .augmentation import Augmentation


class BatchGenerator(Sequence):
    """
    This is a generator for batches of training data. It reads in a selection of images (depending
    on the batch size), augment the images and return them together with target values.
    """

    def __init__(
        self, labeled_data, name, batch_size, input_image_shape, is_grey=False, mask=None, full_rotation_aug=False
    ):
        self.labeled_data = labeled_data
        self.name = name
        self.batch_size = batch_size
        self.input_image_shape = input_image_shape
        self.do_augmentation = name == "train"
        self.augmenter = Augmentation(is_grey,full_rotation=full_rotation_aug)
        self.write_psd = False
        self.do_psd = False
        self.mask = mask

    def __len__(self):
        """
        :return: Return the total number of batches to cover the whole training set.
        """
        num_batches = int(np.ceil(len(self.labeled_data) / self.batch_size))
        return num_batches

    def __getitem__(self, idx):
        """
        :return: Returns the batch idx.
                first output is a numpy array of images
                second output is a list of integer, which represents the label of the images
        """
        if not self.labeled_data:
            print("ERROR: the labeled data set is empty!")
            exit(-1)

        idx = idx % self.__len__()
        start = 0 if idx == 0 else (self.batch_size * idx) - 1
        end = start + self.batch_size
        batch_tubles = self.labeled_data[start:end]

        # Find unique hdf files and read the image
        unique_class_files = {data_tuble[0] for data_tuble in batch_tubles}
        images = []
        psds = []
        labels = []
        for class_file_path in unique_class_files:
            indicis_for_class_file = [
                data_tuble[1]
                for data_tuble in batch_tubles
                if data_tuble[0] == class_file_path
            ]
            labels_for_class_file = [
                data_tuble[2]
                for data_tuble in batch_tubles
                if data_tuble[0] == class_file_path
            ]
            class_index_tubles = [(class_file_path, indi) for indi in indicis_for_class_file]
            images_from_file = getImages_fromList_key(class_index_tubles)

            images_from_file = [
                resize_img(img, (self.input_image_shape[0], self.input_image_shape[1]))
                for img in images_from_file
            ]  # 2. Downsize images to network input size

            images = images + images_from_file

            labels = labels + labels_for_class_file



        if self.do_augmentation is True:

            images = [
                self.augmenter.image_augmentation(img) for img in images
            ]  # 3. Do data augmentation (+ flip image randomly (X,Y,TH, NONE)) but only for training, not for validation

        if self.mask is not None:
            images = [
                apply_mask(img,self.mask) for img in images
            ]

        images = [
            normalize_img(img) for img in images
        ]  # 4. Normalize images ( subtract mean, divide by standard deviation)



        if self.do_psd:
            psds = [calc_2d_spectra(img) for img in images]
            if self.write_psd:
                np.savetxt("psd.txt",psds[0])
            psds = [
                normalize_img(psd) for psd in psds
            ]
            if self.write_psd:
                from tifffile import imsave
                imsave('image.tif', images[0])
                #np.savetxt("image.txt",images[0])
                np.savetxt("psd_norm.txt",psds[0])
                self.write_psd=False

        images = np.array(images)
        if self.do_psd:
            images = np.stack((images, psds), axis=3)
        else:
            images = images[:, :, :, np.newaxis]

        # print(str(idx)+self.name)
        '''
        weights=[]
        for l in labels:
            if l == 0:
                weights.append(self.weight_0)
            elif l == 1:
                weights.append(self.weight_1)
        '''
        return images, labels

    def on_epoch_end(self):
        """
        This method shuffle the training data at the end of a epoch.
        :return: None
        """
        np.random.shuffle(self.labeled_data)


class Auto2DSelectNet:
    """
    Network class for cinderella
    """

    def __init__(self, batch_size, input_size, depth=1, mask_radius=None):
        """

        :param batch_size: Batch size for training / prediction
        :param input_size: Input image size
        """
        self.batch_size = batch_size
        self.input_size = input_size
        self.mask = None
        self.mask_radius = mask_radius
        if mask_radius is not None:
            from .helper import create_circular_mask
            self.mask = create_circular_mask(input_size[0], input_size[1], radius=mask_radius)
        #self.model = self.get_model_unet(input_size=(self.input_size[0],self.input_size[1]))
        self.model = self.build_phosnet_model(depth)

    def build_phosnet_model(self,depth):
        """

        :param input_size: Image input size
        :return: A keras model
        """
        input_image = Input(shape=(self.input_size[0], self.input_size[1], depth))

        # name of first layer
        if depth == 1:
            name_first_layer = "conv_1_depth1"
        else:
            name_first_layer = "bablub"

        # Layer 1
        layer_out = Conv2D(
            32,
            (3, 3),
            strides=(1, 1),
            padding="same",
            name=name_first_layer,
            use_bias=False,
        )(input_image)
        layer_out = BatchNormalization(name="norm_1")(layer_out)
        layer_out = LeakyReLU(alpha=0.1)(layer_out)
        layer_out = MaxPooling2D(pool_size=(2, 2))(layer_out)

        # Layer 2
        layer_out = Conv2D(
            64, (3, 3), strides=(1, 1), padding="same", name="conv_2", use_bias=False
        )(layer_out)
        layer_out = BatchNormalization(name="norm_2")(layer_out)
        layer_out = LeakyReLU(alpha=0.1)(layer_out)
        layer_out = MaxPooling2D(pool_size=(2, 2))(layer_out)

        # Layer 3
        layer_out = Conv2D(
            128, (3, 3), strides=(1, 1), padding="same", name="conv_3", use_bias=False
        )(layer_out)
        layer_out = BatchNormalization(name="norm_3")(layer_out)
        layer_out = LeakyReLU(alpha=0.1)(layer_out)

        # Layer 4
        layer_out = Conv2D(
            64, (1, 1), strides=(1, 1), padding="same", name="conv_4", use_bias=False
        )(layer_out)
        layer_out = BatchNormalization(name="norm_4")(layer_out)
        layer_out = LeakyReLU(alpha=0.1)(layer_out)

        # Layer 5
        layer_out = Conv2D(
            128, (3, 3), strides=(1, 1), padding="same", name="conv_5", use_bias=False
        )(layer_out)
        layer_out = BatchNormalization(name="norm_5")(layer_out)
        layer_out = LeakyReLU(alpha=0.1)(layer_out)
        layer_out = MaxPooling2D(pool_size=(2, 2))(layer_out)

        # Layer 6
        layer_out = Conv2D(
            256, (3, 3), strides=(1, 1), padding="same", name="conv_6", use_bias=False
        )(layer_out)
        layer_out = BatchNormalization(name="norm_6")(layer_out)
        layer_out = LeakyReLU(alpha=0.1)(layer_out)

        # Layer 7
        layer_out = Conv2D(
            128, (1, 1), strides=(1, 1), padding="same", name="conv_7", use_bias=False
        )(layer_out)
        layer_out = BatchNormalization(name="norm_7")(layer_out)
        layer_out = LeakyReLU(alpha=0.1)(layer_out)

        # Layer 8
        layer_out = Conv2D(
            256, (3, 3), strides=(1, 1), padding="same", name="conv_8", use_bias=False
        )(layer_out)
        layer_out = BatchNormalization(name="norm_8")(layer_out)
        layer_out = LeakyReLU(alpha=0.1)(layer_out)
        layer_out = MaxPooling2D(pool_size=(2, 2))(layer_out)

        # Layer 9
        layer_out = Conv2D(
            512, (3, 3), strides=(1, 1), padding="same", name="conv_9", use_bias=False
        )(layer_out)
        layer_out = BatchNormalization(name="norm_9")(layer_out)
        layer_out = LeakyReLU(alpha=0.1)(layer_out)

        # Layer 10
        layer_out = Conv2D(
            256, (1, 1), strides=(1, 1), padding="same", name="conv_10", use_bias=False
        )(layer_out)
        layer_out = BatchNormalization(name="norm_10")(layer_out)
        layer_out = LeakyReLU(alpha=0.1)(layer_out)

        # Layer 11
        layer_out = Conv2D(
            512, (3, 3), strides=(1, 1), padding="same", name="conv_11", use_bias=False
        )(layer_out)
        layer_out = BatchNormalization(name="norm_11")(layer_out)
        layer_out = LeakyReLU(alpha=0.1)(layer_out)

        # Layer 12
        layer_out = Conv2D(
            256, (1, 1), strides=(1, 1), padding="same", name="conv_12", use_bias=False
        )(layer_out)
        layer_out = BatchNormalization(name="norm_12")(layer_out)
        layer_out = LeakyReLU(alpha=0.1)(layer_out)

        # Layer 13
        layer_out = Conv2D(
            512, (3, 3), strides=(1, 1), padding="same", name="conv_13", use_bias=False
        )(layer_out)
        layer_out = BatchNormalization(name="norm_13")(layer_out)
        layer_out = LeakyReLU(alpha=0.1)(layer_out)

        skip_connection = layer_out

        layer_out = MaxPooling2D(pool_size=(2, 2))(layer_out)

        # Layer 14
        layer_out = Conv2D(
            1024, (3, 3), strides=(1, 1), padding="same", name="conv_14", use_bias=False
        )(layer_out)
        layer_out = BatchNormalization(name="norm_14")(layer_out)
        layer_out = LeakyReLU(alpha=0.1)(layer_out)

        # Layer 15
        layer_out = Conv2D(
            512, (1, 1), strides=(1, 1), padding="same", name="conv_15", use_bias=False
        )(layer_out)
        layer_out = BatchNormalization(name="norm_15")(layer_out)
        layer_out = LeakyReLU(alpha=0.1)(layer_out)

        # Layer 16
        layer_out = Conv2D(
            1024, (3, 3), strides=(1, 1), padding="same", name="conv_16", use_bias=False
        )(layer_out)
        layer_out = BatchNormalization(name="norm_16")(layer_out)
        layer_out = LeakyReLU(alpha=0.1)(layer_out)

        # Layer 17
        layer_out = Conv2D(
            512, (1, 1), strides=(1, 1), padding="same", name="conv_17", use_bias=False
        )(layer_out)
        layer_out = BatchNormalization(name="norm_17")(layer_out)
        layer_out = LeakyReLU(alpha=0.1)(layer_out)

        # Layer 18
        layer_out = Conv2D(
            1024, (3, 3), strides=(1, 1), padding="same", name="conv_18", use_bias=False
        )(layer_out)
        layer_out = BatchNormalization(name="norm_18")(layer_out)
        layer_out = LeakyReLU(alpha=0.1)(layer_out)

        # Layer 19
        layer_out = Conv2D(
            1024, (3, 3), strides=(1, 1), padding="same", name="conv_19", use_bias=False
        )(layer_out)
        layer_out = BatchNormalization(name="norm_19")(layer_out)
        layer_out = LeakyReLU(alpha=0.1)(layer_out)

        # Layer 20
        layer_out = Conv2D(
            1024, (3, 3), strides=(1, 1), padding="same", name="conv_20", use_bias=False
        )(layer_out)
        layer_out = BatchNormalization(name="norm_20")(layer_out)
        layer_out = LeakyReLU(alpha=0.1)(layer_out)

        layer_out = UpSampling2D(size=(2, 2))(layer_out)

        skip_connection = Conv2D(
            256, (1, 1), strides=(1, 1), padding="same", name="conv_21_", use_bias=False
        )(skip_connection)
        skip_connection = BatchNormalization(name="norm_21_")(skip_connection)
        skip_connection = LeakyReLU(alpha=0.1)(skip_connection)
        layer_out = concatenate([skip_connection, layer_out])

        # Layer 21
        layer_out = Conv2D(
            1024, (3, 3), strides=(1, 1), padding="same", name="conv_22", use_bias=False
        )(layer_out)
        layer_out = BatchNormalization(name="norm_22")(layer_out)
        layer_out = LeakyReLU(alpha=0.1)(layer_out)

        feature_extractor = Model(input_image, layer_out)

        layer_out = GlobalAveragePooling2D()(feature_extractor(input_image))

        output = Dense(64, activation="relu", name="denseL1")(layer_out)
        # output = Dropout(0.2)(output)
        output = Dense(10, activation="relu", name="denseL2")(output)
        # output = Dropout(0.2)(output)
        output = Dense(1, activation="sigmoid", name="denseL3")(output)

        model = Model(input_image, output)
        model.summary()
        return model

    def get_model_unet(self, kernel_size=(3, 3)):
        inputs = Input(shape=(self.input_size[0], self.input_size[1], 1))
        skips = [inputs]

        x = Conv2D(
            name="enc_conv0",
            filters=48,
            kernel_size=kernel_size,
            padding="same",
            kernel_initializer="he_normal",
        )(inputs)
        x = LeakyReLU(alpha=0.1)(x)
        x = Conv2D(
            name="enc_conv1",
            filters=48,
            kernel_size=kernel_size,
            padding="same",
            kernel_initializer="he_normal",
        )(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D((2, 2))(x)  # --- pool_1
        skips.append(x)

        x = Conv2D(
            name="enc_conv2",
            filters=48,
            kernel_size=kernel_size,
            padding="same",
            kernel_initializer="he_normal",
        )(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D((2, 2))(x)  # --- pool_2
        skips.append(x)

        x = Conv2D(
            name="enc_conv3",
            filters=48,
            kernel_size=kernel_size,
            padding="same",
            kernel_initializer="he_normal",
        )(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D((2, 2))(x)  # --- pool_3
        skips.append(x)

        x = Conv2D(
            name="enc_conv4",
            filters=48,
            kernel_size=kernel_size,
            padding="same",
            kernel_initializer="he_normal",
        )(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D((2, 2))(x)  # --- pool_4
        skips.append(x)

        x = Conv2D(
            name="enc_conv5",
            filters=48,
            kernel_size=kernel_size,
            padding="same",
            kernel_initializer="he_normal",
        )(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D((2, 2))(x)  # --- pool_5 (not re-used)

        x = Conv2D(
            name="enc_conv6",
            filters=48,
            kernel_size=kernel_size,
            padding="same",
            kernel_initializer="he_normal",
        )(x)
        x = LeakyReLU(alpha=0.1)(x)

        x = UpSampling2D((2, 2))(x)

        x = concatenate([x, skips.pop()])
        x = Conv2D(
            name="dec_conv5",
            filters=96,
            kernel_size=kernel_size,
            padding="same",
            kernel_initializer="he_normal",
        )(x)
        x = LeakyReLU(alpha=0.1)(x)

        x = Conv2D(
            name="dec_conv5b",
            filters=96,
            kernel_size=kernel_size,
            padding="same",
            kernel_initializer="he_normal",
        )(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = UpSampling2D((2, 2))(x)

        x = concatenate([x, skips.pop()])

        x = Conv2D(
            name="dec_conv4",
            filters=96,
            kernel_size=kernel_size,
            padding="same",
            kernel_initializer="he_normal",
        )(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Conv2D(
            name="dec_conv4b",
            filters=96,
            kernel_size=kernel_size,
            padding="same",
            kernel_initializer="he_normal",
        )(x)
        x = LeakyReLU(alpha=0.1)(x)

        x = UpSampling2D((2, 2))(x)

        x = concatenate([x, skips.pop()])

        x = Conv2D(
            name="dec_conv3",
            filters=96,
            kernel_size=kernel_size,
            padding="same",
            kernel_initializer="he_normal",
        )(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Conv2D(
            name="dec_conv3b",
            filters=96,
            kernel_size=kernel_size,
            padding="same",
            kernel_initializer="he_normal",
        )(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = UpSampling2D((2, 2))(x)

        x = concatenate([x, skips.pop()])
        x = Conv2D(
            name="dec_conv2",
            filters=96,
            kernel_size=kernel_size,
            padding="same",
            kernel_initializer="he_normal",
        )(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Conv2D(
            name="dec_conv2b",
            filters=96,
            kernel_size=kernel_size,
            padding="same",
            kernel_initializer="he_normal",
        )(x)
        x = LeakyReLU(alpha=0.1)(x)

        x = UpSampling2D((2, 2))(x)
        x = concatenate([x, skips.pop()])

        x = Conv2D(
            name="dec_conv1a",
            filters=64,
            kernel_size=kernel_size,
            padding="same",
            kernel_initializer="he_normal",
        )(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Conv2D(
            name="dec_conv1b",
            filters=32,
            kernel_size=kernel_size,
            padding="same",
            kernel_initializer="he_normal",
        )(x)
        x = LeakyReLU(alpha=0.1)(x)

        outputs = Conv2D(
            filters=1,
            kernel_size=kernel_size,
            padding="same",
            kernel_initializer="he_normal",
        )(x)
        feature_extractor = Model(inputs=inputs, outputs=outputs)

        #layer_out = GlobalAveragePooling2D()(feature_extractor(inputs))
        layer_out = Flatten()(feature_extractor(inputs))

        output = Dense(64, name="denseL1")(layer_out)
        output = LeakyReLU(alpha=0.1)(output)
        # output = Dropout(0.2)(output)
        output = Dense(10, name="denseL2")(output)
        # output = Dropout(0.2)(output)
        output = LeakyReLU(alpha=0.1)(output)
        output = Dense(1, activation="sigmoid", name="denseL3")(output)

        model = Model(inputs, output)
        model.summary()
        return model

    def load_weights(self, model_path):
        """
        Load the weights
        :param model_path: Path to .h5 model file
        :return: None
        """
        self.model.load_weights(model_path, by_name=True)

    # Define our custom loss function

    def train(
        self,
        train_good_path,
        train_bad_path,
        save_weights_name,
        learning_rate=10 ** -4,
        nb_epoch=50,
        nb_epoch_early=10,
        pretrained_weights=None,
        seed=10,
        train_val_thresh=0.8,
        max_valid_img_per_file=10,
        warmrestarts=True,
        valid_good_path=None,
        valid_bad_path=None,
        full_rotation_aug=False
    ):
        """
        Train the network on 2D classes.

        :param train_good_path: Path to folder with good classes
        :param train_bad_path: Path to folder with bad classes
        :param save_weights_name: Filename of the modle file
        :param pretrained_weights: Filepath to pretrained weights
        :param seed: Seed for random number selection
        :return: None
        """
        np.random.seed(seed)

        if os.path.exists(pretrained_weights):
            print("Load pretrained weights", pretrained_weights)
            self.model.load_weights(pretrained_weights, by_name=True,skip_mismatch=True)

        """
        labeled_data = get_data_tubles(good_path, bad_path)
        train_valid_split = int(0.8 * len(labeled_data))

        np.random.shuffle(labeled_data)
        train_data = labeled_data[:train_valid_split]
        valid_data = labeled_data[train_valid_split:]
        """
        if valid_good_path is None and valid_bad_path is None:
            train_data, valid_data, weights = get_train_valid_tubles(
                train_good_path, train_bad_path, train_val_thresh, max_valid_img_per_file
            )
        else:
            train_data, _, weights = get_train_valid_tubles(
                train_good_path, train_bad_path, 1.0
            )
            _, valid_data, _ = get_train_valid_tubles(
                valid_good_path, valid_bad_path, 0.0
            )
        #all_data = valid_data + train_data

        #ccmatrix = get_correlation_matrix(all_data, len(valid_data))
        #print(-np.sort(-1*ccmatrix.flatten()))
        train_generator = BatchGenerator(
            labeled_data=train_data,
            name="train",
            batch_size=self.batch_size,
            input_image_shape=self.input_size,
            mask=self.mask,
            full_rotation_aug=full_rotation_aug
        )
        valid_generator = BatchGenerator(
            labeled_data=valid_data,
            name="valid",
            batch_size=self.batch_size,
            input_image_shape=self.input_size,
            mask=self.mask
        )

        # Define callbacks
        all_callbacks = []
        checkpoint = ModelCheckpoint(
            save_weights_name,
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode="min",
            period=1,
        )
        all_callbacks.append(checkpoint)

        early_stop = EarlyStopping(
            monitor="val_loss",
            min_delta=0.0005,
            patience=nb_epoch_early,
            mode="min",
            verbose=1,
        )
        all_callbacks.append(early_stop)

        reduceLROnPlateau = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.1,
            patience=int(nb_epoch_early * 0.6),
            verbose=1,
        )
        all_callbacks.append(reduceLROnPlateau)

        try:
            os.makedirs(os.path.expanduser("logs/"))
        except:
            pass

        tb_counter = (
            len(
                [
                    log
                    for log in os.listdir(os.path.expanduser("logs/"))
                    if "cinderella" in log
                ]
            )
            + 1
        )
        tensorboard = TensorBoard(
            log_dir=os.path.expanduser("logs/") + "cinderella" + "_" + str(tb_counter),
            histogram_freq=0,
            write_graph=True,
            write_images=False,
        )
        all_callbacks.append(tensorboard)
        if not warmrestarts:
            reduce_lr_on_plateau = ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.1,
                patience=int(nb_epoch_early * 0.6),
                verbose=1,
            )
            all_callbacks.append(reduce_lr_on_plateau)
        else:
            schedule = SGDRScheduler(
                min_lr=1e-7,
                max_lr=1e-4,
                steps_per_epoch=len(train_generator),
                lr_decay=0.9,
                cycle_length=5,
                mult_factor=1.5,
            )
            all_callbacks.append(schedule)
        # define swa callback
        """
        swa = SWA(start_epoch=5,
                  lr_schedule='cyclic',
                  swa_lr=learning_rate*0.1,
                  swa_lr2=learning_rate,
                  swa_freq=4,
                  verbose=1)
        """

        optimizer = Adam(
            lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0
        )

        weights = {1: weights[1], 0: weights[0]}
        #weights = {1: 1, 0: 1}
        print("Weights", weights)
        self.model.compile(
            optimizer=optimizer, metrics=["binary_accuracy"], loss="binary_crossentropy"
        )
        self.model.fit_generator(
            generator=train_generator,
            validation_data=valid_generator,
            workers=12,
            epochs=nb_epoch,
            callbacks=all_callbacks,
            max_queue_size=multiprocessing.cpu_count()//2,
            use_multiprocessing=False,
            class_weight=weights,
        )
        # average_filename = "average.h5"
        # self.model.save_weights(average_filename)

        import h5py

        with h5py.File(save_weights_name, mode="r+") as f:
            f["input_size"] = self.input_size
            if self.mask_radius is not None:
                f["mask_radius"] = [self.mask_radius]

        # with h5py.File(average_filename, mode='r+') as f:
        #    f["input_size"] = self.input_size

        print("Meta data saved in model.")

    def predict(self, input_path, model_path, good_thresh=0.5,invert_images=False):
        """
        Runs a trained model on a .hdf file containing 2D classes.

        :param input_path: Path to input hdf file
        :param model_path: Path to .h5 weights file.
        :param good_thresh: Threshold for selecting good classes
        :return: Return a list of tuples with the format (input_path, index_in_hdf, label, confidence)
        """
        self.load_weights(model_path)

        relevant_files = getList_relevant_files(getList_files(input_path))
        print("Relevant files:", len(relevant_files))
        files_to_classify = []
        for file_to_classify in relevant_files:
            files_to_classify += [(file_to_classify, index) for index in get_key_list_images(file_to_classify) if index != None]

        results = []
        from tqdm import tqdm
        for img_chunk in tqdm(list(chunks(files_to_classify, self.batch_size))):
            list_img = getImages_fromList_key(img_chunk)
            result = self.predict_np_list(list_img, invert_imgs=invert_images,mask=self.mask)
            results.append(result)

        result = np.concatenate(tuple(results))
        result_tuples = []
        for index, img_tuble in enumerate(files_to_classify):
            label = 0
            if result[index] > good_thresh:
                label = 1

            confidence = result[index]
            if result[index] <= good_thresh:
                confidence = 1 - confidence
            result_tuples.append((img_tuble[0], img_tuble[1], label, confidence))
        return result_tuples

    def predict_np_arr(self, images):
        """
        Run the prediction on a 3D with format numpy array [IMDEX_INDEX, IMAGE_WIDTH, IMAGE_HEIGHT].
        :param images: numpy array in the format [IMDEX_INDEX, IMAGE_WIDTH, IMAGE_HEIGHT]
        :return: 1D numpy array with probability of being a good class
        """
        list_img = [images[i] for i in range(images.shape[0])]
        return self.predict_np_list(self, list_img)

    def predict_np_list(self, list_img, invert_imgs=False, mask=None):
        """
        Run the prediction on list of 2d numpy arrays.

        :param list_img: List of 2d numpy arrays (images)
        :return: 1D numpy array with probability of being a good class
        """
        list_img = [
            resize_img(get_relevant_slices(img), (self.input_size[0], self.input_size[1]))
            for img in list_img
        ]  # 2. Downsize images to network input size

        if invert_imgs:
            list_img = [invert(img) for img in list_img]

        if mask is not None:
            list_img = [apply_mask(img,mask) for img in list_img]

        list_img = [normalize_img(img) for img in list_img]



        arr_img = np.array(list_img)
        arr_img = np.expand_dims(arr_img, 3)
        pred_res = self.model.predict(arr_img)
        result = pred_res[:, 0]
        return result

def invert(img):
    img = np.max(img)-img
    return img

def get_relevant_slices(img):
    if len(img.shape) == 2:
        return img
    if len(img.shape) == 3:
        if img.shape[2] == 1:
            return np.squeeze(img)
        else:
            central_slice = int(img.shape[2]/2)
            return img[central_slice,:,:]

def get_data_tubles(good_path, bad_path):
    """
    :param good_path: Path to the folder with good classes
    :param bad_path: Path to the folder with bad classes
    :return: List of tubles (HDF_PATH,CLASS_INDEX,LABEL) LABEL =1 good, 0 bad
    """
    list_bad = list()
    list_good = list()
    for good_p in getList_relevant_files(getList_files(good_path)):
        list_good += [(good_p, index, 1.0) for index in get_key_list_images(good_p)]
    for bad_p in getList_relevant_files(getList_files(bad_path)):
        list_bad += [(bad_p, index, 0.0) for index in get_key_list_images(bad_p)]
    return list_good + list_bad


def get_correlation_matrix(img_list, splitindex=None):
    print("Get COrr", splitindex, len(img_list))
    from scipy import stats
    if splitindex is None:
        cc_matrix = np.zeros(shape=(len(img_list),len(img_list)))
    else:
        cc_matrix = np.zeros(shape=(len(img_list[:splitindex]), len(img_list[splitindex:])))

    if splitindex is None:
        for img_a_index in range(len(img_list)):
            print(img_a_index/len(img_list))
            imga = getImages_fromList_key([(img_list[img_a_index][0], img_list[img_a_index][1])])[0].flatten()
            for img_b_index in range(img_a_index+1,len(img_list)):

                imgb = getImages_fromList_key([(img_list[img_b_index][0],img_list[img_b_index][1])])[0]
                cc = stats.pearsonr(imga, imgb.flatten())[0]
                cc_matrix[img_a_index,img_b_index] = cc
                cc_matrix[img_b_index,img_a_index] = cc
    else:
        for img_a_index in range(0, splitindex):
            print(img_a_index / splitindex)
            imga = getImages_fromList_key([(img_list[img_a_index][0], img_list[img_a_index][1])])[0].flatten()
            for img_b_index in range(splitindex, len(img_list)):
                imgb = getImages_fromList_key([(img_list[img_b_index][0], img_list[img_b_index][1])])[0]
                cc = stats.pearsonr(imga, imgb.flatten())[0]
                cc_matrix[img_a_index,img_b_index-splitindex] = cc
    return cc_matrix

def get_train_valid_tubles(good_path, bad_path, thresh=0.9, max_val_img_per_file=-1):
    list_bad_train = list()
    list_good_train = list()
    list_bad_valid = list()
    list_good_valid = list()

    for good_p in getList_relevant_files(getList_files(good_path)):
        good_tubles = [(good_p, index, 1.0) for index in get_key_list_images(good_p)]
        if len(good_tubles)>1:
            train_valid_split = int(thresh * len(good_tubles))
            if max_val_img_per_file > -1:
                train_valid_split = max(
                    train_valid_split, len(good_tubles) - max_val_img_per_file
                )
            np.random.shuffle(good_tubles)

            # Do the valid/train split for each file
            list_good_train += good_tubles[:train_valid_split]
            list_good_valid += good_tubles[train_valid_split:]
        else:
            if np.random.rand() > thresh:
                list_good_valid.extend(good_tubles)
            else:
                list_good_train.extend(good_tubles)

    for bad_p in getList_relevant_files(getList_files(bad_path)):
        bad_tubles = [(bad_p, index, 0.0) for index in get_key_list_images(bad_p)]
        if len(bad_tubles) > 1:
            train_valid_split = int(thresh * len(bad_tubles))
            if max_val_img_per_file > -1:
                train_valid_split = max(
                    train_valid_split, len(bad_tubles) - max_val_img_per_file
                )
            np.random.shuffle(bad_tubles)

            list_bad_train += bad_tubles[:train_valid_split]
            list_bad_valid += bad_tubles[train_valid_split:]
        else:
            if np.random.rand() > thresh:
                list_bad_valid.extend(bad_tubles)
            else:
                list_bad_train.extend(bad_tubles)
    print("CLASS 1", len(list_good_train),"(+",len(list_good_valid)," val. img)", "CLASS 0", len(list_bad_train),"(+",len(list_bad_valid),"val img.)")

    list_train = list_good_train + list_bad_train
    list_valid = list_good_valid + list_bad_valid
    weight_good=0
    weight_bad=0
    if len(list_train)>0:
        if len(list_good_train) > len(list_bad_train):
            weight_good = 1
            weight_bad = len(list_good_train)/len(list_bad_train)
        else:
            weight_good = len(list_bad_train) / len(list_good_train)
            weight_bad = 1

        print("Class weight 1:", weight_good, "Class weight 0:", weight_bad)
    return list_train, list_valid, (weight_bad, weight_good)


def chunks(list_to_divide, number_of_chunks):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(list_to_divide), number_of_chunks):
        yield list_to_divide[i : i + number_of_chunks]


def focal_loss(y_true, y_pred):
    """Focal loss for multi-classification
    FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
    Notice: y_pred is probability after softmax
    gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
    d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
    Focal Loss for Dense Object Detection
    https://arxiv.org/abs/1708.02002

    Arguments:
        y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
        y_pred {tensor} -- model's output, shape of [batch_size, num_cls]

    Keyword Arguments:
        gamma {float} -- (default: {2.0})
        alpha {float} -- (default: {4.0})

    Returns:
        [tensor] -- loss.
    """
    import tensorflow as tf

    gamma = 2.0
    alpha = 4.0
    epsilon = 1.0e-9
    y_true = tf.convert_to_tensor(y_true, tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, tf.float32)

    model_out = tf.add(y_pred, epsilon)
    ce = tf.multiply(y_true, -tf.log(model_out))
    weight = tf.multiply(y_true, tf.pow(tf.subtract(1.0, model_out), gamma))
    fl = tf.multiply(alpha, tf.multiply(weight, ce))
    reduced_fl = tf.reduce_max(fl, axis=1)
    return tf.reduce_mean(reduced_fl)


def focal_loss(y_true, y_pred):
    import tensorflow as tf

    gamma = 2.0
    alpha = 4.0
    # epsilon = 1.e-9
    # y_pred = tf.add(y_pred, epsilon)
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1.0 - pt_1, gamma) * K.log(pt_1)) - K.sum(
        alpha * K.pow(pt_0, gamma) * K.log(1.0 - pt_0)
    )
