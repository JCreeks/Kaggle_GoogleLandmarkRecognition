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
from utils.data_util import save_obj, load_obj, make_ids_list, load_model_from_file
from utils.pred_util import add_unknown_imgs, pred_generator, split_seq, square_error, gap
from conf.configure import *
from conf.generatorConf import *
from conf.predConf import *

if __name__ == "__main__":
    test = True
    train_lim = 0#2**15
    pieces = 20
    whole_ids_list = make_ids_list(test_images_folder)
    if train_lim:
        whole_ids_list = whole_ids_list[:train_lim]
    total_ids = len(whole_ids_list)
        
#     xcpetion1 = load_model('../xception/1st_phase_xcpetion_model.h5')
#     xcpetion2 = load_model('../xception/second_second_phase_logs/1525082681_xcpetion_model.h5')
    xcpetion1_0 = '../xception/first_phase_logs/1525550924_xcpetion_model.h5'
    xcpetion1_1 = '../xception/first_phase_logs/1525597648_xcpetion_model.h5'
    xcpetion1_2 = '../xception/first_phase_logs/1525640444_xcpetion_model.h5'
    xcpetion2 = '../xception/2nd_phase_xcpetion_model.h5'
    xcpetion3 = '../xception/3rd_phase_xcpetion_model.h5'
    inceptRes3 = '../inceptionResnet/3rd_phase_inceptionResnet_model.h5'
        
    pathList = [inceptRes3]
    modelList = []
    for path in pathList:
        modelList.append(load_model(path))

    total = len(split_seq(whole_ids_list, int(total_ids / pieces)))
    with open(results_file, 'w') as f:
        f.write('id,landmarks')
    
    predProbList = []
    for counter, ids_list in enumerate(split_seq(whole_ids_list, int(total_ids / pieces))):
        steps = int(len(ids_list) / pred_batch_size)
        steps += 0 if len(ids_list) % pred_batch_size == 0 else 1
        
        predList=[]
        for model in modelList:            
#             model = load_model(model)
            pred_list = ids_list[:]
            pred_gen = pred_generator(pred_list, test_images_folder, pred_batch_size, input_shape, normalize=True)
            pred = model.predict_generator(pred_gen, steps=steps, verbose=2)
            predList.append(pred)
            del pred_gen
        
        predicts = np.ones(predList[0].shape)
        for pred in predList:
            predicts*=pred
        predicts**=(1./len(predList))
        certainties = np.max(predicts, axis=-1)
        labels_inds = np.argmax(predicts, axis=-1)

        with open(results_file, 'a') as f:
            text = ''
            for i in range(len(ids_list)):
                text += "\n" + str(ids_list[i]) + "," + inverted_class_indices_dict[labels_inds[i]] + " " +\
                        str(certainties[i])
            f.write(text)
            
        for i, pred in enumerate(predList):
            if len(predProbList)<len(modelList):
                predProbList.append(pred)
            else:  
                predProbList[i]=np.append(predProbList[i], pred, axis=0)

        print('done {} out of {}'.format(counter + 1, total))

    if test:
        add_unknown_imgs(results_file)
        
    ensResultPath = '../ensemble/result'
    if not os.path.exists(ensResultPath):
        os.makedirs(ensResultPath)
    for i, path in enumerate(pathList):
        np.save(os.path.join(ensResultPath, path.split('/')[-1]+'.npy'), predProbList[i])