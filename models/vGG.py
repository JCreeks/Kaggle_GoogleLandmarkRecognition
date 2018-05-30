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
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_resnet_v2 import InceptionResNetV2
# from keras.applications.densenet import DenseNet201
from keras.applications.vgg19 import VGG19

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
from utils.pred_util import square_error, gap, GapCallback, set_reg_drop, GAP_vector
from conf.configure import *
from conf.generatorConf import *
from preprocess.generator import make_generators

def first_phase(trained=True, printGap=True, first_phase_train_reps=first_phase_train_reps):
    global VGG_model
    tensorboard = TensorBoard(log_dir=first_phase_folder + 'tb_logs', batch_size=batch_size)

    if not trained:
        # create the base pre-trained model
        input_tensor = Input(shape=input_shape)
        base_model = VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False)

        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # add a fully-connected layer
        x = Dense(1024, activation='relu')(x)
        # add a logistic layer
        predictions = Dense(classes_num, activation='softmax')(x)

        # this is the model we will train
        VGG_model = Model(inputs=base_model.input, outputs=predictions)
        VGG_model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['acc'])
    else:
        VGG_model = load_model(data_folder + '1st_phase_VGG_model.h5')
        
    if not os.path.exists(first_phase_folder):
        os.makedirs(first_phase_folder)
        
    for i in range(first_phase_train_reps):
        history = VGG_model.fit_generator(train_img_class_gen,
                                     steps_per_epoch=steps_per_small_epoch,
                                     epochs=small_epochs, 
                                     verbose=2,
                                     validation_data=val_img_class_gen, validation_steps=val_steps_per_small_epoch,
                                     workers=4, callbacks=[tensorboard])
        print('itr', i)
        if i % saves_per_epoch == 0:
            print('{} epoch completed'.format(int(i / saves_per_epoch)))
        
        if i>=5:
            ts = calendar.timegm(time.gmtime())
            VGG_model.save(first_phase_folder + str(ts) + '_VGG_model.h5')
            save_obj(history.history, str(ts) + '_VGG_history', folder=first_phase_folder)
        
        if printGap:
            steps = len(val_names_list)/batch_size
            predicts = VGG_model.predict_generator(val_img_class_gen, steps=steps/10, verbose=2)##########
            predProb = np.max(predicts, axis=-1)
            predId = np.argmax(predicts, axis=-1)
            trueId = list(map(lambda x: val_name_id_dict[str(x).split('.')[0].split('/')[1]], [name for name in val_img_class_gen.filenames]))
            gap = GAP_vector(predId, predProb, trueId)
            print('gap: ', gap)

        VGG_model.save(data_folder + '1st_phase_VGG_model.h5')
    
def VGG(trained=False, third_phase_train_reps=third_phase_train_reps):
    global VGG_model, new_VGG_model, optimizer
    tensorboard = TensorBoard(log_dir=third_phase_folder + 'tb_logs', batch_size=batch_size)
    start_lr = 0.00015
    end_lr = 0.00001
    step_lr = (end_lr - start_lr) / (third_phase_train_reps - 1)
    
    if not trained:
        # create the base pre-trained model
        input_tensor = Input(shape=input_shape)
        base_model = InceptionResNetV2(input_tensor=input_tensor, weights='imagenet', include_top=False)

        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # add a fully-connected layer
        x = Dense(1024, activation='relu')(x)
        # add a logistic layer
        predictions = Dense(classes_num, activation='softmax')(x)

        # this is the model we will train
        VGG_model = Model(inputs=base_model.input, outputs=predictions)
        VGG_model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['acc'])
        
        for layer in VGG_model.layers:
            layer.trainable = True
            if isinstance(layer, keras.layers.convolutional.Conv2D):
                layer.kernel_regularizer = regularizers.l2(0.001)
                layer.activity_regularizer = regularizers.l1(0.001)  
        
        # add dropout and regularizer to the penultimate Dense layer
        predictions = VGG_model.layers[-1]
        dropout = Dropout(0.2)
        fc = VGG_model.layers[-2]
        fc.kernel_regularizer = regularizers.l2(0.001)
        fc.activity_regularizer = regularizers.l1(0.001)

        x = dropout(fc.output)
        predictors = predictions(x)
        new_VGG_model = Model(inputs=VGG_model.input, outputs=predictors)

        optimizer = Adam(lr=0.1234)
        
        new_VGG_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
    else:
        new_VGG_model = load_model(data_folder + '3rd_phase_VGG_model.h5') 
        optimizer = Adam(lr=0.1234)
        new_VGG_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])

    if not os.path.exists(third_phase_folder):
        os.makedirs(third_phase_folder)
        
    for i in range(third_phase_train_reps):
        lr = start_lr + step_lr * i
        K.set_value(new_VGG_model.optimizer.lr, lr)
        print(i, 'out of ', third_phase_train_reps, '\nlearning rate ', K.eval(new_VGG_model.optimizer.lr))
        history = new_VGG_model.fit_generator(train_img_class_gen,
                                               steps_per_epoch=steps_per_small_epoch,
                                               epochs=small_epochs, verbose=2,
                                               validation_data=val_img_class_gen, validation_steps=val_steps_per_small_epoch,
                                               workers=4, callbacks=[tensorboard])
#         history = new_VGG_model.fit_generator(train_img_class_gen,
#                                                    steps_per_epoch=steps_per_small_epoch,
#                                                    epochs=small_epochs, verbose=2,
#                                                    validation_data=val_img_class_gen, validation_steps=val_steps_per_small_epoch,
#                                                    workers=4, callbacks=[LosswiseKerasCallback(tag='keras VGG model')])
        print("iteration",i)
        if i % saves_per_epoch == 0:
            print('{} epoch completed'.format(int(i / saves_per_epoch)))
        
        if i>=5:
            ts = calendar.timegm(time.gmtime())
            new_VGG_model.save(third_phase_folder + str(ts) + '_VGG_model.h5')
            save_obj(history.history, str(ts) + '_VGG_history.h5', folder=third_phase_folder)

    new_VGG_model.save(data_folder + '3rd_phase_VGG_model.h5')
    
    
def third_phase(trained=False, third_phase_train_reps=third_phase_train_reps):
    global VGG_model, new_VGG_model, optimizer
    tensorboard = TensorBoard(log_dir=third_phase_folder + 'tb_logs', batch_size=batch_size)
    
    if not trained:
        VGG_model = load_model(data_folder + '1st_phase_VGG_model.h5')
    else:
        VGG_model = load_model(data_folder + '3rd_phase_VGG_model.h5')

#     # add regularizers to the convolutional layers
#     trainable_layers_ratio = 1 / 2.0
#     trainable_layers_index = int(len(VGG_model.layers) * (1 - trainable_layers_ratio))
#     for layer in VGG_model.layers[:trainable_layers_index]:
#         layer.trainable = False
#     for layer in VGG_model.layers[trainable_layers_index:]:
#         layer.trainable = True

    for layer in VGG_model.layers:
        layer.trainable = True
        if isinstance(layer, keras.layers.convolutional.Conv2D):
            layer.kernel_regularizer = regularizers.l2(0.001)
            layer.activity_regularizer = regularizers.l1(0.001)

    # add dropout and regularizer to the penultimate Dense layer
    predictions = VGG_model.layers[-1]
    dropout = Dropout(0.2)
    fc = VGG_model.layers[-2]
    fc.kernel_regularizer = regularizers.l2(0.001)
    fc.activity_regularizer = regularizers.l1(0.001)

    x = dropout(fc.output)
    predictors = predictions(x)
    new_VGG_model = Model(inputs=VGG_model.input, outputs=predictors)

    optimizer = Adam(lr=0.1234)
    start_lr = 0.0001
    end_lr = 0.0001
    step_lr = (end_lr - start_lr) / (third_phase_train_reps - 1)
    new_VGG_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])

    if not os.path.exists(third_phase_folder):
        os.makedirs(third_phase_folder)
        
    for i in range(third_phase_train_reps):
        lr = start_lr + step_lr * i
        K.set_value(new_VGG_model.optimizer.lr, lr)
        print(i, 'out of ', third_phase_train_reps, '\nlearning rate ', K.eval(new_VGG_model.optimizer.lr))
        history = new_VGG_model.fit_generator(train_img_class_gen,
                                               steps_per_epoch=steps_per_small_epoch,
                                               epochs=small_epochs, verbose=2,
                                               validation_data=val_img_class_gen, validation_steps=val_steps_per_small_epoch,
                                               workers=4, callbacks=[tensorboard])
#         history = new_VGG_model.fit_generator(train_img_class_gen,
#                                                    steps_per_epoch=steps_per_small_epoch,
#                                                    epochs=small_epochs, verbose=2,
#                                                    validation_data=val_img_class_gen, validation_steps=val_steps_per_small_epoch,
#                                                    workers=4, callbacks=[LosswiseKerasCallback(tag='keras VGG model')])
        print("iteration",i)
        if i % saves_per_epoch == 0:
            print('{} epoch completed'.format(int(i / saves_per_epoch)))
        
        if i>=5:
            ts = calendar.timegm(time.gmtime())
            new_VGG_model.save(third_phase_folder + str(ts) + '_VGG_model.h5')
            save_obj(history.history, str(ts) + '_VGG_history.h5', folder=third_phase_folder)

    new_VGG_model.save(data_folder + '3rd_phase_VGG_model.h5')
    
if __name__ == '__main__':
    
    train_img_class_gen, val_img_class_gen=make_generators(isSimple=True)
#     iVGG(trained=True)
#    first_phase(trained=True, printGap=False, first_phase_train_reps=10)
#     second_phase()
    third_phase(trained=False, third_phase_train_reps=4)
#     second_second_phase(trained=True)
#     continue_second(trained=True)
