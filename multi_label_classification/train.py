import sys
sys.path.append('./*')

import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
import os
from data import generate_train_val
from model import build_googlenet

cur_dir = os.getcwd()
datasets_dir = os.path.join(cur_dir, 'datasets')
data_dir = os.path.join(datasets_dir, 'sport')

IMG_SIZE = 80
BATCH_SIZE = 16
EPOCH = 50
NUM_CLASS = 30
AUTOTUNE = tf.data.AUTOTUNE
init = 'he_normal'
regul = keras.regularizers.l2(2e-4)
DATA_EPOCH = 2
seed_list = [135, 214]

train_ds, val_ds = generate_train_val(DATA_EPOCH, seed_list)

googlenet = build_googlenet(IMG_SIZE*3, NUM_CLASS, regul, init)

googlenet.summary()

learning_rates = [2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
learning_rate_boundaries = [125, 250, 500, 240000, 360000]
learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=learning_rate_boundaries, values=learning_rates
)

optimizers = Adam(learning_rate=learning_rate_fn)

googlenet.compile(
    loss="binary_crossentropy",
    optimizer=optimizers,
    metrics=['accuracy']
)

filename = 'checkpoint-epoch-{}.h5'.format(EPOCH)
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
earlystopping = EarlyStopping(monitor='val_loss', patience=20, mode='min')

googlenet.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCH,
    verbose=1,
    callbacks=[checkpoint, earlystopping]
)

googlenet.evaluate(val_ds)