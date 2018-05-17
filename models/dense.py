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

from keras.models import Sequential, Model, load_model
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import GlobalAveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.layers import Dense, Input, Flatten, Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.applications import VGG16#, DenseNet121
from keras import optimizers
from keras.layers.normalization import BatchNormalization
import argparse
from os.path import join
from scipy import misc
import numpy as np
from skimage import transform

import losswise
from losswise.libs import LosswiseKerasCallback
losswise.set_api_key('JWN8A6X96')

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)
from utils.data_util import save_obj, load_obj
from utils.pred_util import square_error, gap, GapCallback, set_reg_drop, RocAucMetricCallback
from conf.configure import *
from conf.generatorConf import *
from preprocess.generator import make_generators

def get_model(classes):
    model = Sequential()

    model.add(Convolution2D(64, (3,3), activation="relu", input_shape=(None, None, 3)))
    model.add(Convolution2D(64, (3,3), activation="relu", padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Convolution2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Convolution2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Convolution2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Convolution2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Convolution2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Convolution2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Convolution2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Convolution2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Convolution2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Convolution2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Convolution2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(GlobalMaxPooling2D())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(classes, activation='softmax'))

#     model.summary()

    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,
            loss='categorical_crossentropy',
            metrics=['accuracy'])

    return model

def modified_pretrained_model(classes, pretrained_weights="DenseNet121", freeze_top=False):

    if pretrained_weights == "DenseNet121":
        pretrained_conv_model = DenseNet121(weights="imagenet", include_top=False)
        
        if freeze_top:
            for layer in pretrained_conv_model.layers:
                layer.trainable = False

        input = Input(shape=(None, None, 3),name = 'image_input')
        output_pretrained_conv = pretrained_conv_model(input)

        eps = 1.1e-5
        final_stage = "final"
        x = BatchNormalization(epsilon=eps, axis=3, name='conv'+str(final_stage)+'_blk_bn')(output_pretrained_conv)
        x = Activation('relu', name='relu'+str(final_stage)+'_blk')(x)
        x = GlobalAveragePooling2D(name='pool'+str(final_stage))(x)
        x = Dense(classes, name='predictions')(x)
        x = Activation('softmax', name='prob')(x)

    elif pretrained_weights == "VGG16":
        pretrained_conv_model = VGG16(weights='imagenet', include_top=False)

        if freeze_top:
            for layer in pretrained_conv_model.layers:
                layer.trainable = False

        input = Input(shape=(None, None, 3),name = 'image_input')
        output_pretrained_conv = pretrained_conv_model(input)

        x = GlobalAveragePooling2D()(output_pretrained_conv)
        x = Dense(512, activation='relu', name='fc1')(x)
        x = Dense(256, activation='relu', name='fc2')(x)
        x = Dense(classes, activation="softmax")(x)
        
    model = Model(input=input, output=x)
#     model.summary()

    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,
            loss='categorical_crossentropy',
            metrics=['accuracy'])
    
    return model

def dense(trained=True, argsModel="VGG16", argsFreeze=False):
    reg_num = 0.00
    drop_rate = 0.0
    print("reg_num: ", reg_num, " drop_rate:", drop_rate)
    
    if not os.path.exists(dense_folder):
        os.makedirs(dense_folder)

    if not trained:
        model = get_model(classes_num)
        modelID = "dense"
    else:
        model = modified_pretrained_model(
            classes_num,
            pretrained_weights=argsModel,
            freeze_top=argsFreeze)
        modelID = "dense_pretrained"
        
    model = set_reg_drop(model, reg_num, drop_rate)

    if not os.path.exists(dense_phase_folder):
        os.makedirs(dense_phase_folder)
    
    best_model = model
    for i in range(dense_phase_train_reps):
        print(i + 1, 'out of ', dense_phase_train_reps)

#         monitor = EarlyStopping('roc_auc_val',patience=20, verbose=2)
        monitor = EarlyStopping(monitor='val_acc', min_delta=1e-3, patience=5, verbose=0, mode='auto')
        ts = calendar.timegm(time.gmtime())
        checkpointer = ModelCheckpoint(filepath=dense_phase_folder+str(ts)+'best_weights.hdf5', verbose=0, save_best_only=True) # save best model

        model.fit_generator(train_img_class_gen,
                                       steps_per_epoch=steps_per_small_epoch,
                                       epochs=small_epochs, verbose=2,
                                       validation_data=val_img_class_gen,
                                       validation_steps=val_steps_per_small_epoch,
                                       workers=4,
                                       callbacks=[#RocAucMetricCallback(),
                                           monitor,checkpointer])

        if i % saves_per_epoch == 0:
            print('{} epoch completed'.format(int(i / saves_per_epoch)))
        
        best_model.load_weights(dense_phase_folder+str(ts)+'best_weights.hdf5')
        best_model.save(dense_phase_folder+str(ts)+'best_models.h5')
        
    best_model.save(dense_folder + modelID + '.h5')
    
if __name__ == '__main__':
    
    train_img_class_gen, val_img_class_gen=make_generators(isSimple=True)
    dense(trained=False)
#     dense(trained=True)