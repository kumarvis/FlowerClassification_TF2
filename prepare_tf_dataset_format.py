import os
import pathlib
import numpy as np
import tensorflow as tf
import functools
import project_params
from KV_Utils.file_utils import img_name_helper

AUTOTUNE = tf.data.experimental.AUTOTUNE
CLASS_NAMES = None

def get_class_name(base_folder_path):
    data_dir = pathlib.Path(base_folder_path)
    result = [item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"]
    result.sort()
    return result

def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    lbl = tf.cast((parts[-2] == CLASS_NAMES), tf.int16)
    lbl = tf.argmax(lbl, axis=0)
    return lbl

def decode_img(img, augmentation):
    # convert the compressed string to a 3D uint8 tensor
    if project_params.img_ext == 'jpg':
        img = tf.image.decode_jpeg(img, channels=3)
    if project_params.img_ext == 'png':
        img = tf.image.decode_png(img, channels=3)

    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    if augmentation:
        image = tf.image.random_flip_left_right(img)
    # resize the image to the desired size.
    img_wd, img_ht = project_params.img_size, project_params.img_size
    return tf.image.resize(img, [img_wd, img_ht])

def _prepare_data_flower_classificaion(file_path, augmentation):
    label = get_label(file_path)
    # load the raw data from the file as a string
    image = tf.io.read_file(file_path)
    image = decode_img(image, augmentation)
    return image, label

def get_dataset_ready(data_dir_path, mode = 'Train', augment=False):
    data_dir = pathlib.Path(data_dir_path)
    global CLASS_NAMES
    CLASS_NAMES = np.asarray(get_class_name(data_dir))
    no_images = len(img_name_helper.get_img_list(data_dir_path, project_params.img_ext))
    shuffle_buffer_size = no_images
    batch_sz = project_params.exp_batch_sz
    if augment:
        prepare_data_flower_classificaion = functools.partial(_prepare_data_flower_classificaion, augmentation=True)
    else:
        prepare_data_flower_classificaion = functools.partial(_prepare_data_flower_classificaion, augmentation=False)

    list_ds = tf.data.Dataset.list_files(str(data_dir / '*/*'))
    labeled_ds = list_ds.shuffle(buffer_size=shuffle_buffer_size)

    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    labeled_ds = list_ds.map(prepare_data_flower_classificaion, num_parallel_calls=AUTOTUNE)

    # Repeat forever
    labeled_ds = labeled_ds.repeat()
    #Batch Dataset
    labeled_ds = labeled_ds.batch(batch_size=batch_sz)
    # `prefetch` lets the dataset fetch batches in the background while the model is training.
    labeled_ds = labeled_ds.prefetch(buffer_size=AUTOTUNE)
    return labeled_ds, no_images

