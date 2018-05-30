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

from keras.layers import Input, Conv2D, Dropout, merge, Dense, Flatten, MaxPooling2D, GlobalAveragePooling2D, InputLayer
from keras import layers
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
from utils.siamese_util import set_trainables
from conf.configure import *
from conf.generatorConf import *
from preprocess.generator import make_siamese_generators

def SiameseModel(base_model_path):
#     siamese_model = load_model_from_file(data_folder + 'siamese_model_.json')
#     siamese_model.load_weights(data_folder + '2nd_phase_siamese_weights.h5', by_name=False)
    siamese_model=load_model(base_model_path)
    siamese_model.get_layer('dense_1').activation = K.sigmoid

    short_xception_model = Model(inputs=siamese_model.input, outputs=siamese_model.get_layer('dense_1').output)
    short_xception_model = set_trainables(short_xception_model, ('dense_1'))

#     print(short_xception_model.summary())

    left_input = Input(input_shape)
    right_input = Input(input_shape)

    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        encoded_l = short_xception_model(left_input)
        encoded_r = short_xception_model(right_input)

    difference = layers.subtract([encoded_l, encoded_r])
    merged = layers.multiply([difference, difference])
    prediction = Dense(1, activation='sigmoid', name='prediction_dense', trainable=True)(merged)
    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)

    return siamese_net


def first_phase(trained=True, printGap=True, first_phase_train_reps=first_phase_train_reps, data_folder=data_folder):
    global siamese_model
    tensorboard = TensorBoard(log_dir=first_phase_folder + 'tb_logs', batch_size=batch_size)

    if not trained:
        siamese_model = SiameseModel('../resnet/' + '3rd_phase_resnet_model.h5')
        
        optimizer = Adam(0.0005)
        siamese_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['acc'])
    else:
        siamese_model = load_model(data_folder + '1st_phase_siamese_model.h5')
        
    if not os.path.exists(first_phase_folder):
        os.makedirs(first_phase_folder)
        
    for i in range(first_phase_train_reps):
        history = siamese_model.fit_generator(train_img_class_gen,
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
            siamese_model.save(first_phase_folder + str(ts) + '_siamese_model.h5')
            save_obj(history.history, str(ts) + '_siamese_history', folder=first_phase_folder)
        
        if printGap:
            steps = len(val_names_list)/batch_size
            predicts = siamese_model.predict_generator(val_img_class_gen, steps=steps/10, verbose=2)##########
            predProb = np.max(predicts, axis=-1)
            predId = np.argmax(predicts, axis=-1)
            trueId = list(map(lambda x: val_name_id_dict[str(x).split('.')[0].split('/')[1]], [name for name in val_img_class_gen.filenames]))
            gap = GAP_vector(predId, predProb, trueId)
            print('gap: ', gap)

        siamese_model.save(data_folder + '1st_phase_siamese_model.h5')
    
if __name__ == '__main__':
    
    train_img_class_gen, val_img_class_gen=make_siamese_generators()
    first_phase(trained=False, printGap=False, first_phase_train_reps=5)
#     second_phase()
#     third_phase(third_phase_train_reps=10)
#     second_second_phase(trained=True)
#     continue_second(trained=True)
