"""

The labeled data is organized in the following manner.

Your program gets two folders:
    1. GOOD_CLASSES
    2. BAD_CLASSES

Each folder can contain multiple hdf files in subfolders. Each hdf file contains multiple images (classes).
HDF 0: CLASS 1, CLASS 2, CLASS 3, ..., CLASS N_0
HDF 1: CLASS 1, CLASS 2, CLASS 3, ..., CLASS N_1
...
HDF K: CLASS 1, CLASS 2, CLASS 3, ..., CLASS N_K

You first have to divide your data into training and validation data.
Therefore I would organize the data as a list of tubles (HDF_PATH,CLASS_INDEX,LABEL):

labled_data_shuffled = [(HDF_PATH_0,1,LABEL_0),(HDF_PATH_0,2,LABEL_0),(HDF_PATH_0,3,LABEL_0),...,(HDF_PATH_0,N_0,LABEL_0),
                        (HDF_PATH_1,1,LABEL_1),(HDF_PATH_1,2,LABEL_1),(HDF_PATH_1,3,LABEL_1),...(HDF_PATH_1,N_1,LABEL_1),...,
                        (HDF_PATH_K,N_0,LABEL_K),(HDF_PATH_K,1,LABEL_K),(HDF_PATH_K,2,LABEL_K),(HDF_PATH_K,3,LABEL_K),...(HDF_PATH_K,N_1,LABEL_K)]
You can then shuffle this list and select randomly 20% as validation data and 80% as training data.


Finally you have 3 lists:
train_data = 80% of labeled_data_shuffled
valid_data = 20% of labeled_data_shuffled

A generator is used to provide the data for training/


"""

import os
from keras.utils import Sequence
from keras.layers import (
    Conv2D,
    Input,
    MaxPooling2D,
    BatchNormalization,
    Dense,
    Flatten,
    UpSampling2D,
    GlobalAveragePooling2D,
)
from keras.layers.merge import concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
import multiprocessing
import numpy as np
import time
from .helper import (
    getImages_fromList_key,
    resize_img,
    normalize_img,
    get_list_images,
    getList_hdf_files,
    getList_files,
)
from .augmentation import Augmentation


class BatchGenerator(Sequence):
    """
    This is a generator for batches of training data. It reads in a selection of images (depending
    on the batch size), augment the images and return them together with target values.
    """

    def __init__(self, labled_data, name, batch_size, input_image_shape, is_grey=False):
        self.labled_data = labled_data
        self.name = name
        self.batch_size = batch_size
        self.input_image_shape = input_image_shape
        self.do_augmentation = name == "train"
        self.augmenter = Augmentation(is_grey)

    def __len__(self):
        """
        :return: Return the total number of batches to cover the whole training set.
        """
        num_batches = int(np.ceil(len(self.labled_data) / self.batch_size))
        return num_batches

    def __getitem__(self, idx):
        """
        :return: Returns the batch idx.
                first output is a numpy array of images
                second output is a list of integer, which represents the label of the images
        """
        if len(self.labled_data) == 0:
            print("ERROR: the labeled data set is empty!")
            exit(-1)

        """ select the set of images"""
        idx = idx % self.__len__()
        start = 0 if idx == 0 else (self.batch_size * idx) - 1
        end = start + self.batch_size
        batch_tubles = self.labled_data[start:end]

        # Find unique hdf files and read the image
        unique_hdfs = set()
        [unique_hdfs.add(data_tuble[0]) for data_tuble in batch_tubles]
        x = []
        y = []
        for hdf_path in unique_hdfs:
            indicis_for_hdf = [
                data_tuble[1]
                for data_tuble in batch_tubles
                if data_tuble[0] == hdf_path
            ]
            labels_for_hdf = [
                data_tuble[2]
                for data_tuble in batch_tubles
                if data_tuble[0] == hdf_path
            ]
            x = x + getImages_fromList_key(hdf_path, indicis_for_hdf)
            y = y + labels_for_hdf

        x = [
            resize_img(img, (self.input_image_shape[0], self.input_image_shape[1]))
            for img in x
        ]  # 2. Downsize images to network input size
        if self.do_augmentation is True:
            x = [
                self.augmenter.image_augmentation(img) for img in x
            ]  # 3. Do data augmentation (+ flip image randomly (X,Y,TH, NONE)) but only for training, not for validation
        x = [
            normalize_img(img) for img in x
        ]  # 4. Normalize images ( subtract mean, divide by standard deviation)
        x = np.array(x)
        x = x[:, :, :, np.newaxis]
        # print(str(idx)+self.name)
        return x, y

    def on_epoch_end(self):
        """
        This method shuffle the training data at the end of a epoch.
        :return: None
        """
        np.random.shuffle(self.labled_data)


class Auto2DSelectNet(object):
    def __init__(self, batch_size, input_size):
        self.batch_size = batch_size
        self.input_size = input_size
        self.model = self.build_phosnet_model(self.input_size)

    def build_phosnet_model(self, input_size):
        input_image = Input(shape=(input_size[0], input_size[1], 1))

        # name of first layer
        name_first_layer = "conv_1_depth1"

        # Layer 1
        x = Conv2D(
            32,
            (3, 3),
            strides=(1, 1),
            padding="same",
            name=name_first_layer,
            use_bias=False,
        )(input_image)
        x = BatchNormalization(name="norm_1")(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 2
        x = Conv2D(
            64, (3, 3), strides=(1, 1), padding="same", name="conv_2", use_bias=False
        )(x)
        x = BatchNormalization(name="norm_2")(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 3
        x = Conv2D(
            128, (3, 3), strides=(1, 1), padding="same", name="conv_3", use_bias=False
        )(x)
        x = BatchNormalization(name="norm_3")(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 4
        x = Conv2D(
            64, (1, 1), strides=(1, 1), padding="same", name="conv_4", use_bias=False
        )(x)
        x = BatchNormalization(name="norm_4")(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 5
        x = Conv2D(
            128, (3, 3), strides=(1, 1), padding="same", name="conv_5", use_bias=False
        )(x)
        x = BatchNormalization(name="norm_5")(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 6
        x = Conv2D(
            256, (3, 3), strides=(1, 1), padding="same", name="conv_6", use_bias=False
        )(x)
        x = BatchNormalization(name="norm_6")(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 7
        x = Conv2D(
            128, (1, 1), strides=(1, 1), padding="same", name="conv_7", use_bias=False
        )(x)
        x = BatchNormalization(name="norm_7")(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 8
        x = Conv2D(
            256, (3, 3), strides=(1, 1), padding="same", name="conv_8", use_bias=False
        )(x)
        x = BatchNormalization(name="norm_8")(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 9
        x = Conv2D(
            512, (3, 3), strides=(1, 1), padding="same", name="conv_9", use_bias=False
        )(x)
        x = BatchNormalization(name="norm_9")(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 10
        x = Conv2D(
            256, (1, 1), strides=(1, 1), padding="same", name="conv_10", use_bias=False
        )(x)
        x = BatchNormalization(name="norm_10")(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 11
        x = Conv2D(
            512, (3, 3), strides=(1, 1), padding="same", name="conv_11", use_bias=False
        )(x)
        x = BatchNormalization(name="norm_11")(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 12
        x = Conv2D(
            256, (1, 1), strides=(1, 1), padding="same", name="conv_12", use_bias=False
        )(x)
        x = BatchNormalization(name="norm_12")(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 13
        x = Conv2D(
            512, (3, 3), strides=(1, 1), padding="same", name="conv_13", use_bias=False
        )(x)
        x = BatchNormalization(name="norm_13")(x)
        x = LeakyReLU(alpha=0.1)(x)

        skip_connection = x

        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 14
        x = Conv2D(
            1024, (3, 3), strides=(1, 1), padding="same", name="conv_14", use_bias=False
        )(x)
        x = BatchNormalization(name="norm_14")(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 15
        x = Conv2D(
            512, (1, 1), strides=(1, 1), padding="same", name="conv_15", use_bias=False
        )(x)
        x = BatchNormalization(name="norm_15")(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 16
        x = Conv2D(
            1024, (3, 3), strides=(1, 1), padding="same", name="conv_16", use_bias=False
        )(x)
        x = BatchNormalization(name="norm_16")(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 17
        x = Conv2D(
            512, (1, 1), strides=(1, 1), padding="same", name="conv_17", use_bias=False
        )(x)
        x = BatchNormalization(name="norm_17")(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 18
        x = Conv2D(
            1024, (3, 3), strides=(1, 1), padding="same", name="conv_18", use_bias=False
        )(x)
        x = BatchNormalization(name="norm_18")(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 19
        x = Conv2D(
            1024, (3, 3), strides=(1, 1), padding="same", name="conv_19", use_bias=False
        )(x)
        x = BatchNormalization(name="norm_19")(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 20
        x = Conv2D(
            1024, (3, 3), strides=(1, 1), padding="same", name="conv_20", use_bias=False
        )(x)
        x = BatchNormalization(name="norm_20")(x)
        x = LeakyReLU(alpha=0.1)(x)

        x = UpSampling2D(size=(2, 2))(x)

        skip_connection = Conv2D(
            256, (1, 1), strides=(1, 1), padding="same", name="conv_21_", use_bias=False
        )(skip_connection)
        skip_connection = BatchNormalization(name="norm_21_")(skip_connection)
        skip_connection = LeakyReLU(alpha=0.1)(skip_connection)
        x = concatenate([skip_connection, x])

        # Layer 21
        x = Conv2D(
            1024, (3, 3), strides=(1, 1), padding="same", name="conv_22", use_bias=False
        )(x)
        x = BatchNormalization(name="norm_22")(x)
        x = LeakyReLU(alpha=0.1)(x)

        feature_extractor = Model(input_image, x)

        x = GlobalAveragePooling2D()(feature_extractor(input_image))
        # output = Flatten()(x)
        output = Dense(64, activation="relu", name="denseL1")(x)
        output = Dense(10, activation="relu", name="denseL2")(output)
        output = Dense(1, activation="sigmoid", name="denseL3")(output)

        model = Model(input_image, output)
        model.summary()
        return model

    def get_data_tubles(self, good_path, bad_path):
        """
        :param good_path: Path to the folder with good classes
        :param bad_path: Path to the folder with bad classes
        :return: List of tubles (HDF_PATH,CLASS_INDEX,LABEL) LABEL =1 good, 0 bad
        """
        list_bad = list()
        list_good = list()
        for good_p in getList_hdf_files(getList_files(good_path)):
            list_good += [(good_p, index, 1.0) for index in get_list_images(good_p)]
        for bad_p in getList_hdf_files(getList_files(bad_path)):
            list_bad += [(bad_p, index, 0.0) for index in get_list_images(bad_p)]
        return list_good + list_bad

    def train(
        self, good_path, bad_path, save_weights_name, pretrained_weights=None, seed=10
    ):
        np.random.seed(10)

        if os.path.exists(pretrained_weights):
            print("Load pretrained weights", pretrained_weights)
            self.model.load_weights(pretrained_weights, by_name=True)

        labeled_data = self.get_data_tubles(good_path, bad_path)
        train_valid_split = int(0.8 * len(labeled_data))
        np.random.shuffle(labeled_data)
        train_data = labeled_data[:train_valid_split]
        valid_data = labeled_data[train_valid_split:]

        train_generator = BatchGenerator(
            labled_data=train_data,
            name="train",
            batch_size=self.batch_size,
            input_image_shape=self.input_size,
        )
        valid_generator = BatchGenerator(
            labled_data=valid_data,
            name="valid",
            batch_size=self.batch_size,
            input_image_shape=self.input_size,
        )

        # Define callbacks
        checkpoint = ModelCheckpoint(
            save_weights_name,
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode="min",
            period=1,
        )

        early_stop = EarlyStopping(
            monitor="val_loss", min_delta=0.0005, patience=10, mode="min", verbose=1
        )
        optimizer = Adam(
            lr=10 ** -4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0
        )
        self.model.compile(
            optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
        )
        self.model.fit_generator(
            generator=train_generator,
            validation_data=valid_generator,
            workers=multiprocessing.cpu_count() // 2,
            epochs=50,
            callbacks=[checkpoint, early_stop],
            max_queue_size=multiprocessing.cpu_count(),
            use_multiprocessing=False,
        )

    def predict(self, input_path, model_path, good_thresh=0.5):
        self.model.load_weights(model_path)
        img_list = get_list_images(input_path)
        results = []
        from tqdm import tqdm

        for img_chunk in tqdm(list(self.chunks(img_list, self.batch_size))):
            list_img = getImages_fromList_key(input_path, img_chunk)
            list_img = [
                resize_img(img, (self.input_size[0], self.input_size[1]))
                for img in list_img
            ]  # 2. Downsize images to network input size
            list_img = [normalize_img(img) for img in list_img]
            arr_img = np.array(list_img)
            arr_img = np.expand_dims(arr_img, 3)

            pred_res = self.model.predict(arr_img)
            result = pred_res[:, 0]
            results.append(result)

        result = np.concatenate(tuple(results))
        result_tuples = []
        for index, index_in_hdf in enumerate(img_list):
            label = 0
            if result[index] > good_thresh:
                label = 1

            confidence = result[index]
            if result[index] <= good_thresh:
                confidence = 1 - confidence
            result_tuples.append((input_path, index_in_hdf, label, confidence))

        return result_tuples

    def chunks(self, l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i : i + n]
