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

IMG_SIZE = 80
BATCH_SIZE = 64
EPOCHS = 100
NUM_CLASS = 30
AUTOTUNE = tf.data.AUTOTUNE

class_lsit = os.listdir(data_dir)




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

    train_dataset, train_label = np.array(train_dataset) / 255., np.array(train_label)
    val_dataset, val_label = np.array(val_dataset) / 255., np.array(val_label)

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







train_dataset = []
train_label = []
val_dataset = []
val_label = []
seed = [542,123,543,123,885,512,157,852,170]
for idx in range(9):
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
        ran_idx = np.random.randint(3, 10)
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
        ran_idx = np.random.randint(3, 10)
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

train_dataset, train_label = np.array(train_dataset) / 255., np.array(train_label)
val_dataset, val_label = np.array(val_dataset) / 255., np.array(val_label)

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

from tensorflow.keras.applications import EfficientNetB7

model = EfficientNetB7(include_top=False, weights='imagenet', input_shape=(240, 240, 3))

for layer in model.layers[:-30]:
    layer.trainable = False

from keras.layers import Dropout, Dense, Conv2D, MaxPooling2D, Activation, Concatenate
from keras.layers import Flatten, Input, LayerNormalization, AveragePooling2D, GlobalAveragePooling2D
from keras.models import Model
from tensorflow.keras.optimizers import Adam

from tensorflow_addons.metrics import F1Score

x = model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(30, activation='sigmoid')(x)

model = tf.keras.models.Model(inputs=model.input, outputs=output)



learning_rates = [2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
learning_rate_boundaries = [125, 250, 500, 240000, 360000]
learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=learning_rate_boundaries, values=learning_rates
)

optimizers = Adam(learning_rate=learning_rate_fn)


model.compile(optimizer=optimizers,
              loss='binary_crossentropy',
              metrics=['accuracy', F1Score(num_classes=30, average='micro')])

model.fit(
    final_train,
    validation_data=final_val,
    epochs=50,
    verbose=1,
    #callbacks=[checkpoint, earlystopping]
)