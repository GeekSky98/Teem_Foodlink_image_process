import sys
sys.path.append('./*')

import tensorflow as tf
from data import generate_train_val
from keras.layers import Dropout, Dense, GlobalAveragePooling2D
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB7
from tensorflow_addons.metrics import F1Score
import os

cur_dir = os.getcwd()
datasets_dir = os.path.join(cur_dir, 'datasets')
data_dir = os.path.join(datasets_dir, 'sport')

IMG_SIZE = 80
BATCH_SIZE = 64
EPOCH = 50
NUM_CLASS = 30
DATA_EPOCH = 9
seed_list = [135, 214, 153, 654, 138, 632, 316, 687, 125]

train_ds, val_ds = generate_train_val(DATA_EPOCH, seed_list)

model = EfficientNetB7(include_top=False, weights='imagenet', input_shape=(IMG_SIZE*3, IMG_SIZE*3, 3))

for layer in model.layers[:-30]:
    layer.trainable = False

x = model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(NUM_CLASS, activation='sigmoid')(x)

model = Model(inputs=model.input, outputs=output)

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
    train_ds,
    validation_data=val_ds,
    epochs=EPOCH,
    verbose=1,
    #callbacks=[checkpoint, earlystopping]
)