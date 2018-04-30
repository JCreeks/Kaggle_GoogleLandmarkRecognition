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
from keras.layers import Input, Conv2D, Dropout, merge, Dense, Flatten, MaxPooling2D, GlobalAveragePooling2D, InputLayer
from keras.models import Model, Sequential
from keras import backend as K
from keras.models import load_model, model_from_json
from keras.optimizers import Adam
K.image_data_format() == 'channels_last'
from keras.utils import generic_utils
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras.callbacks import TensorBoard, Callback
from keras.losses import categorical_hinge, mean_squared_error, mean_absolute_error, categorical_crossentropy
import keras

import losswise
from losswise.libs import LosswiseKerasCallback
losswise.set_api_key('JWN8A6X96')

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)
from utils.data_util import save_obj, load_obj
from conf.configure import *
from conf.generatorConf import *
from preprocess.generator import make_generators

def first_phase():
    global xcpetion_model
    tensorboard = TensorBoard(log_dir=first_phase_folder + 'tb_logs', batch_size=batch_size)
    # create the base pre-trained model
    input_tensor = Input(shape=input_shape)
    base_model = Xception(input_tensor=input_tensor, weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # add a logistic layer
    predictions = Dense(classes_num, activation='softmax')(x)
    #predictions = Dense(train_classes_num, activation='softmax')(x)
    
    # this is the model we will train
    xcpetion_model = Model(inputs=base_model.input, outputs=predictions)
    xcpetion_model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['acc'])

    for i in range(first_phase_train_reps):
        history = xcpetion_model.fit_generator(train_img_class_gen,
                                     steps_per_epoch=steps_per_small_epoch,
                                     epochs=small_epochs, verbose=2,
                                     validation_data=val_img_class_gen, validation_steps=val_steps_per_small_epoch,
                                     workers=4, callbacks=[tensorboard])
        print(i)
        if i % saves_per_epoch == 0:
            print('{} epoch completed'.format(int(i / saves_per_epoch)))

        ts = calendar.timegm(time.gmtime())
        xcpetion_model.save(first_phase_folder + str(ts) + '_xcpetion_model.h5')
        save_obj(history.history, str(ts) + '_xcpetion_history', folder=first_phase_folder)

    xcpetion_model.save(data_folder + '1st_phase_xcpetion_model.h5')
    
def second_phase():
    global xcpetion_model
    tensorboard = TensorBoard(log_dir=second_phase_folder + 'tb_logs', batch_size=batch_size)
    xcpetion_model = load_model(data_folder + '1st_phase_xcpetion_model.h5')

    trainable_layers_ratio = 1/3.0
    trainable_layers_index = int(len(xcpetion_model.layers) * (1 - trainable_layers_ratio))
    for layer in xcpetion_model.layers[:trainable_layers_index]:
       layer.trainable = False
    for layer in xcpetion_model.layers[trainable_layers_index:]:
       layer.trainable = True

    # for layer in xcpetion_model.layers:
    #     layer.trainable = True

    xcpetion_model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['acc'])

    # train the model on the new data for a few epochs
    for i in range(second_phase_train_reps):
        history = xcpetion_model.fit_generator(train_img_class_gen,
                                               steps_per_epoch=steps_per_small_epoch,
                                               epochs=small_epochs, verbose=2,
                                               validation_data=val_img_class_gen, validation_steps=val_steps_per_small_epoch,
                                               workers=4, callbacks=[tensorboard])
        print(i)
        if i % saves_per_epoch == 0:
            print('{} epoch completed'.format(int(i / saves_per_epoch)))

        ts = calendar.timegm(time.gmtime())
        xcpetion_model.save(second_phase_folder + str(ts) + '_xcpetion_model.h5')
        save_obj(history.history, str(ts) + '_xcpetion_history.h5', folder=second_phase_folder)

    xcpetion_model.save(data_folder + '2nd_phase_xcpetion_model.h5')
    
def third_phase():
    global xcpetion_model, new_xcpetion_model, optimizer
    tensorboard = TensorBoard(log_dir=third_phase_folder + 'tb_logs', batch_size=batch_size)
    xcpetion_model = load_model(data_folder + '2nd_phase_xcpetion_model.h5')

    # add regularizers to the convolutional layers
    trainable_layers_ratio = 1 / 2.0
    trainable_layers_index = int(len(xcpetion_model.layers) * (1 - trainable_layers_ratio))
    for layer in xcpetion_model.layers[:trainable_layers_index]:
        layer.trainable = False
    for layer in xcpetion_model.layers[trainable_layers_index:]:
        layer.trainable = True

    for layer in xcpetion_model.layers:
        layer.trainable = True
        if isinstance(layer, keras.layers.convolutional.Conv2D):
            layer.kernel_regularizer = regularizers.l2(0.001)
            layer.activity_regularizer = regularizers.l1(0.001)

    # add dropout and regularizer to the penultimate Dense layer
    predictions = xcpetion_model.layers[-1]
    dropout = Dropout(0.3)
    fc = xcpetion_model.layers[-2]
    fc.kernel_regularizer = regularizers.l2(0.001)
    fc.activity_regularizer = regularizers.l1(0.001)

    x = dropout(fc.output)
    predictors = predictions(x)
    new_xcpetion_model = Model(inputs=xcpetion_model.input, outputs=predictors)

    optimizer = Adam(lr=0.1234)
    start_lr = 0.001
    end_lr = 0.00001
    step_lr = (end_lr - start_lr) / (third_phase_train_reps - 1)
    new_xcpetion_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])

    for i in range(third_phase_train_reps):
        lr = start_lr + step_lr * i
        K.set_value(new_xcpetion_model.optimizer.lr, lr)
        print(i, 'out of ', third_phase_train_reps, '\nlearning rate ', K.eval(new_xcpetion_model.optimizer.lr))
        history = new_xcpetion_model.fit_generator(train_img_class_gen,
                                               steps_per_epoch=steps_per_small_epoch,
                                               epochs=small_epochs, verbose=2,
                                               validation_data=val_img_class_gen, validation_steps=val_steps_per_small_epoch,
                                               workers=4, callbacks=[tensorboard])
#         history = new_xcpetion_model.fit_generator(train_img_class_gen,
#                                                    steps_per_epoch=steps_per_small_epoch,
#                                                    epochs=small_epochs, verbose=2,
#                                                    validation_data=val_img_class_gen, validation_steps=val_steps_per_small_epoch,
#                                                    workers=4, callbacks=[LosswiseKerasCallback(tag='keras xcpetion model')])
        print("iteration",i)
        if i % saves_per_epoch == 0:
            print('{} epoch completed'.format(int(i / saves_per_epoch)))

        ts = calendar.timegm(time.gmtime())
        new_xcpetion_model.save(third_phase_folder + str(ts) + '_xcpetion_model.h5')
        save_obj(history.history, str(ts) + '_xcpetion_history.h5', folder=third_phase_folder)

    new_xcpetion_model.save(data_folder + '3rd_phase_xcpetion_model.h5')

def second_second_phase(trained=True):
    global xcpetion_model, new_xcpetion_model
    #dropout_Callback = Dropout_Callback()
    tensorboard = TensorBoard(log_dir=second_second_phase_folder + 'tb_logs', batch_size=batch_size)

    if not trained:
        xcpetion_model = load_model(data_folder + '2nd_phase_xcpetion_model.h5')

        # add dropout and regularizer to the penultimate Dense layer

        predictions = xcpetion_model.layers[-1]
        dropout = Dropout(0.2)
        fc = xcpetion_model.layers[-2]
        x = dropout(fc.output)
        predictors = predictions(x)
        new_xcpetion_model = Model(inputs=xcpetion_model.input, outputs=predictors)
    else:
        xcpetion_model = load_model(data_folder + '2nd_2nd_phase_xcpetion_model.h5')
        new_xcpetion_model = xcpetion_model
    
    lr = 0.00005#0.0001
    new_xcpetion_model.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy', metrics=['acc'])

    # train the model on the new data for a few epochs
    for i in range(second_phase_train_reps):
#         history = new_xcpetion_model.fit_generator(train_img_class_gen,
#                                                steps_per_epoch= steps_per_small_epoch,
#                                                epochs=small_epochs, verbose=2,
#                                                validation_data=val_img_class_gen, validation_steps=val_steps_per_small_epoch,
#                                                workers=4, callbacks=[tensorboard, dropout_Callback])
        history = new_xcpetion_model.fit_generator(train_img_class_gen,
                                               steps_per_epoch= steps_per_small_epoch,
                                               epochs=small_epochs, verbose=2,
                                               validation_data=val_img_class_gen, validation_steps=val_steps_per_small_epoch,
                                               workers=4, callbacks=[tensorboard])
        print("iteration", i)
        if i % saves_per_epoch == 0:
            print('{} epoch completed'.format(int(i / saves_per_epoch)))

        ts = calendar.timegm(time.gmtime())
        new_xcpetion_model.save(second_second_phase_folder + str(ts) + '_xcpetion_model.h5')
        save_obj(history.history, str(ts) + '_xcpetion_history.h5', folder=second_second_phase_folder)

    new_xcpetion_model.save(data_folder + '2nd_2nd_phase_xcpetion_model.h5')
        
if __name__ == '__main__':
    
    train_img_class_gen, val_img_class_gen=make_generators(isSimple=False)
#     first_phase()
#     second_phase()
#     third_phase()
    second_second_phase(trained=True)