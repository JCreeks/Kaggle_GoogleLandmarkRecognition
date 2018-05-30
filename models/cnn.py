import pickle
# from cv2 import imread, resize
# from PIL import Image
from pathlib import Path
import random
import calendar
import time
import os, sys

import warnings
warnings.filterwarnings('ignore')

from matplotlib.pyplot import imshow
import numpy as np
from PIL import Image

import numpy as np
import tensorflow as tf
from sklearn.metrics import label_ranking_average_precision_score
from keras.applications.xception import Xception
from keras.layers import Activation, Input, Conv2D, Dropout, merge, Dense, Flatten, MaxPooling2D, GlobalAveragePooling2D, InputLayer
from keras.models import Model, Sequential
from keras import backend as K
from keras.models import load_model, model_from_json
from keras.optimizers import Adam
K.image_data_format() == 'channels_last'
from keras.utils import generic_utils
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras.callbacks import TensorBoard, Callback, EarlyStopping, ModelCheckpoint
from keras.losses import categorical_hinge, mean_squared_error, mean_absolute_error, categorical_crossentropy
import keras

import losswise
from losswise.libs import LosswiseKerasCallback
losswise.set_api_key('JWN8A6X96')

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)
from utils.data_util import save_obj, load_obj
from utils.pred_util import square_error, gap, GapCallback, set_reg_drop
from conf.configure import *
from conf.generatorConf import *
from preprocess.generator import make_generators

def get_model(lr = 1e-3, lr_d = 0):
    model = Sequential()
    model.add(Conv2D(128, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dense(14951, activation="softmax"))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr = lr, decay = lr_d), metrics=['acc'])
#     model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
    return model

def cnn(trained=True):
    reg_num = 0.00
    drop_rate = 0.0
    print("reg_num: ", reg_num, " drop_rate:", drop_rate)
    
    if not os.path.exists(cnn_folder):
        os.makedirs(cnn_folder)

    if not trained:
        model = get_model()
    else:
        model = load_model(cnn_folder + 'cnn_first_phase.h5')
#         model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc']) 
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr = 1e-3, decay = 0), metrics=['acc'])
        
    model = set_reg_drop(model, reg_num, drop_rate)

    if not os.path.exists(cnn_first_phase_folder):
        os.makedirs(cnn_first_phase_folder)
    
    best_model = get_model()
    for i in range(cnn_first_phase_train_reps):
        print(i + 1, 'out of ', cnn_first_phase_train_reps)
        
        monitor = EarlyStopping(monitor='val_acc', min_delta=1e-3, patience=20, verbose=0, mode='auto')
        ts = calendar.timegm(time.gmtime())
        checkpointer = ModelCheckpoint(filepath=cnn_first_phase_folder+str(ts)+'best_weights.hdf5', verbose=0, save_best_only=True) # save best model

        model.fit_generator(train_img_class_gen,
                                       steps_per_epoch=steps_per_small_epoch,
                                       epochs=small_epochs, verbose=2,
                                       validation_data=val_img_class_gen,
                                       validation_steps=val_steps_per_small_epoch,
                                       workers=4,
                                       callbacks=[monitor,checkpointer])

        if i % saves_per_epoch == 0:
            print('{} epoch completed'.format(int(i / saves_per_epoch)))
        
        best_model.load_weights(cnn_first_phase_folder+str(ts)+'best_weights.hdf5')
        best_model.save(cnn_first_phase_folder+str(ts)+'best_models.h5')
        
    best_model.save(cnn_folder + 'cnn_first_phase.h5')
    
if __name__ == '__main__':
    
    train_img_class_gen, val_img_class_gen=make_generators(isPlain=True)
    cnn(trained=False)