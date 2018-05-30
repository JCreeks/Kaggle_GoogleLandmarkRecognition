
# coding: utf-8

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
import time

import numpy as np
# import tensorflow as tf
# from sklearn.metrics import label_ranking_average_precision_score
# from keras.applications.xception import Xception
# from keras.layers import Input, Conv2D, Dropout, merge, Dense, Flatten, MaxPooling2D, GlobalAveragePooling2D, InputLayer
# from keras.models import Model, Sequential
# from keras import backend as K
# from keras.models import load_model, model_from_json
# from keras.optimizers import Adam
# K.image_data_format() == 'channels_last'
# from keras.utils import generic_utils
# from keras.preprocessing.image import ImageDataGenerator
# from keras import regularizers
# from keras.callbacks import TensorBoard, Callback
# from keras.losses import categorical_hinge, mean_squared_error, mean_absolute_error, categorical_crossentropy
# import keras

# import losswise
# from losswise.libs import LosswiseKerasCallback
# losswise.set_api_key('JWN8A6X96')

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)
from utils.data_util import save_obj, load_obj, make_ids_list, load_model_from_file
from utils.pred_util import add_unknown_imgs, pred_generator, split_seq, square_error, gap, GAP_vector
from conf.configure import *
from conf.generatorConf import *
from conf.predConf import *


def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-k', '--k', default=5, help='top k prob')
#     args = parser.parse_args()

#     k = args.k
    k = 1
    
    xcpetion1_0 = '1525550924_xcpetion_model.h5'#.029
    xcpetion1_1 = '1525597648_xcpetion_model.h5'#.059
    xcpetion1_2 = '1525640444_xcpetion_model.h5'#.046
    xcpetion2 = '2nd_phase_xcpetion_model.h5'#.051
    xcpetion3 = '3rd_phase_xcpetion_model.h5'#.064
    resnet1 = '1st_phase_resnet_model.h5'#.047
    resnet3 = '3rd_phase_resnet_model.h5'#.081 #0.082
    inceptRes1 = '1st_phase_inceptionResnet_model.h5'#.050
    inceptRes3 = '3rd_phase_inceptionResnet_model.h5'#.064 0.066
    vgg3 = '3rd_phase_VGG_model.h5'#.027
    dense1 = '1st_phase_denseNet_model.h5'#.083 #0.084
    dense3 = '3rd_phase_denseNet_model.h5'#.089
#     inception1 = '1st_phase_inception_model.h5'#.083
    inception3 = '3rd_phase_inception_model.h5'#.065 #0.0670
    dense169_1 = '1st_phase_denseNet169_model.h5'#.068 #0.070
    dense169_3 = '3rd_phase_denseNet169_model.h5'#.110   
    dense121_3 = '3rd_phase_denseNet121_model.h5'#.083  
    # pathList = [xcpetion1_0,xcpetion1_1,xcpetion1_2,xcpetion2,xcpetion3]
    pathList = [
                resnet3, 
    #             xcpetion3, 
    #             xcpetion2,
    #             xcpetion1_1,
    #             inceptRes3,
    #             vgg3,
                dense1,
                dense3,
    #             inception3,
    #             dense169_1,
                dense169_3,
                dense121_3,
                ]

    resultPath = './val_result/'
    
    predProbList = []
    for i, path in enumerate(pathList):
        predProbList.append(np.load(resultPath+path+'.npy'))
        print(i)
        
    y_true = load_obj('val_ids_list', '../ensemble/')
    
    WList = [3.5,2.,4.,4.5,3.5]
    
    def printGap(WList):        
        assert len(WList)==len(pathList)

        predicts = np.zeros((121861, 14951))
        for pred in predProbList:
            predicts+=(pred**8)*WList[i]
    #        predicts=np.maximum(predicts,np.load(resultPath+path+'.npy'))
    #     predicts*=(1./sum(WList))
    #     certainties = np.sort(predicts, axis=-1)[:,-k:]
    #     labels_inds = np.argsort(predicts, axis=-1)[:,-k:]
        certainties = np.max(predicts, axis=-1)
        labels_inds = np.argmax(predicts, axis=-1)
        del predicts

        y_pred = [int(inverted_class_indices_dict[ind]) for ind in labels_inds]
        conf = certainties

        for path, w in zip(pathList, WList):
            print(path.split('_')[2],w)
        print('gap: {:.4f}'.format(GAP_vector(y_pred, conf, y_true, return_x=False)))
        print('################\n\n')
    
    W = WList
    for w0 in [3.5]:
        W[0]=w0
        for w1 in [2.,3.]:
            W[1]=w1
            for w2 in [4.,3.,2.]:
                W[2]=w2
                for w3 in [4.5,6,4,3.5]:
                    W[3]=w3
                    for w4 in [3.5]:
                        W[4]=w4
                        printGap(W)

if __name__ == "__main__":
    main()

