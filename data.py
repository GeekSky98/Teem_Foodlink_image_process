import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import random
import math
# import shutil

cur_dir = os.getcwd()
datasets_dir = os.path.join(cur_dir, 'datasets')
data_dir = os.path.join(datasets_dir, 'sport')

IMG_SIZE = 200
BATCH_SIZE = 64
EPOCHS = 100
NUM_CLASS = 30
AUTOTUNE = tf.data.AUTOTUNE

class_lsit = os.listdir(data_dir)

# Change Image Extensions
'''
for classes in class_lsit:
    image_dir = os.path.join(data_dir, classes)
    image_files = [file_name for file_name in os.listdir(image_dir) if os.path.splitext(file_name)[-1] != '.jpg']
    print(image_files)
    for image in image_files:
        image_path = os.path.join(image_dir, image)
        new_image_path = os.path.splitext(image_path)[0] + '.jpg'
        os.rename(image_path, new_image_path)
'''

# Used when train / validation directory separation is required
'''
train_dir = os.path.join(datasets_dir, 'train')
val_dir = os.path.join(datasets_dir, 'validation')
split_ratio = 0.9

if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(val_dir):
    os.makedirs(val_dir)

target_list = [train_dir, val_dir]
for classes in class_lsit:
    for target in target_list:
        target_dir = os.path.join(target, classes)
        os.mkdir(target_dir)


for class_dir in class_lsit:
    class_path = os.path.join(data_dir, class_dir)

    if not os.path.isdir(class_path):
        continue

    image_list = os.listdir(class_path)
    split_idx = int(len(image_list) * split_ratio)

    train_images = image_list[:split_idx]
    val_images = image_list[split_idx:]

    train_dir_class = os.path.join(train_dir, class_dir)
    for image in train_images:
        train_img_path = os.path.join(class_path, image)
        train_dir_path = os.path.join(train_dir_class, image)
        shutil.copyfile(train_img_path, train_dir_path)

    val_dir_class = os.path.join(val_dir, class_dir)
    for image in val_images:
        val_img_path = os.path.join(class_path, image)
        val_dir_path = os.path.join(val_dir_class, image)
        shutil.copyfile(val_img_path, val_dir_path)
'''


def generate_train_val(epoch, seed, min_object=3):
    train_dataset = []
    train_label = []
    val_dataset = []
    val_label = []
    assert len(seed) == epoch, "seed_list Error"

    for idx in range(epoch):
        train_ds = keras.preprocessing.image_dataset_from_directory(
            directory=data_dir,
            label_mode='categorical',
            shuffle=True,
            validation_split=0.1,
            subset='training',
            image_size=(IMG_SIZE, IMG_SIZE),
            batch_size=9,
            seed=seed[idx]
        )
        for image, label in train_ds.take(len(train_ds)-1):
            ran_idx = np.random.randint(min_object, 10)
            ran_list = random.sample(range(0, 9), 9-ran_idx)
            image, label = np.array(image), np.array(label)
            if len(ran_list) > 0:
                for i in ran_list:
                    image[i] = np.array(tf.random.uniform(
                        shape=(IMG_SIZE, IMG_SIZE, 3),minval=0, maxval=255, dtype=tf.float32
                    ))
                    label[i] = np.zeros(shape=(NUM_CLASS,))

            row_1 = np.concatenate([image[0], image[1], image[2]], axis=1)
            row_2 = np.concatenate([image[3], image[4], image[5]], axis=1)
            row_3 = np.concatenate([image[6], image[7], image[8]], axis=1)
            merged_image = np.concatenate([row_1, row_2, row_3], axis=0)
            merged_label = tf.reduce_sum(label, axis=0)
            merged_label = tf.clip_by_value(merged_label, clip_value_min=0, clip_value_max=1)

            train_dataset.append(merged_image)
            train_label.append(merged_label)

        val_ds = keras.preprocessing.image_dataset_from_directory(
            directory=data_dir,
            label_mode='categorical',
            validation_split=0.1,
            subset='validation',
            image_size=(IMG_SIZE, IMG_SIZE),
            batch_size=9,
            seed=seed[idx]
        )
        for image, label in val_ds.take(len(val_ds)-1):
            ran_idx = np.random.randint(min_object, 10)
            ran_list = random.sample(range(0, 9), 9-ran_idx)
            image, label = np.array(image), np.array(label)
            if len(ran_list) > 0:
                for i in ran_list:
                    image[i] = np.array(tf.random.uniform(
                        shape=(IMG_SIZE, IMG_SIZE, 3),minval=0, maxval=255, dtype=tf.float32
                    ))
                    label[i] = np.zeros(shape=(NUM_CLASS,))

            row_1 = np.concatenate([image[0], image[1], image[2]], axis=1)
            row_2 = np.concatenate([image[3], image[4], image[5]], axis=1)
            row_3 = np.concatenate([image[6], image[7], image[8]], axis=1)
            merged_image = np.concatenate([row_1, row_2, row_3], axis=0)
            merged_label = tf.reduce_sum(label, axis=0)
            merged_label = tf.clip_by_value(merged_label, clip_value_min=0, clip_value_max=1)

            val_dataset.append(merged_image)
            val_label.append(merged_label)

        train_dataset, train_label = np.array(train_dataset), np.array(train_label)
        val_dataset, val_label = np.array(val_dataset), np.array(val_label)

    total_train = tf.data.Dataset.from_tensor_slices((train_dataset, train_label))
    total_val = tf.data.Dataset.from_tensor_slices((val_dataset, val_label))

    data_augmentation = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode="nearest"
    )
    train_aug = data_augmentation.flow(train_dataset, train_label, batch_size=BATCH_SIZE)

    for _ in range(math.ceil(len(train_dataset)/BATCH_SIZE)):
        aug_data_batch = tf.data.Dataset.from_tensor_slices(next(train_aug))
        total_train = total_train.concatenate((aug_data_batch))

    buf_size = len(total_train) * 2
    final_train = total_train.batch(BATCH_SIZE, drop_remainder=True).shuffle(buffer_size=buf_size).prefetch(AUTOTUNE)
    final_val = total_val.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    return final_train, final_val