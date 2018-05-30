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
from sklearn.metrics.pairwise import cosine_similarity

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
from scipy.spatial.distance import cosine

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
dense3 = '3rd_phase_denseNet_model.h5'#.081 #0.082
#     inception1 = '1st_phase_inception_model.h5'#.083
inception3 = '3rd_phase_inception_model.h5'#.065 #0.0670
dense169_1 = '1st_phase_denseNet169_model.h5'#.068 #0.070
dense169_3 = '3rd_phase_denseNet169_model.h5'#.110 
dense121_3 = '3rd_phase_denseNet121_model.h5'#.083
nasnet3 = '3rd_phase_nasnet_model.h5'#.066

# pathList = [xcpetion1_0,xcpetion1_1,xcpetion1_2,xcpetion2,xcpetion3]
pathList = [
            nasnet3,
            dense121_3,
            dense169_3,
            dense169_1,
            inception3,
            dense3,
            resnet3, 
            xcpetion2,
            xcpetion3, 
            inceptRes3,
            vgg3,
            dense1,
           ]

resultPath = './result/'

def meanCosSim(arr1, arr2):
    assert(arr1.shape==arr2.shape)
    out=0.
    for i in np.arange(arr1.shape[0]):
        out+=1-cosine(arr1[i,:],arr2[i,:])
    return out/arr1.shape[0]

# index=np.random.randint(total_ids, size=size)
n=len(pathList)
for i in [0]:
    for j in np.arange(i+1,n):
        narr1=np.load(resultPath+pathList[i]+'.npy')
        narr2=np.load(resultPath+pathList[j]+'.npy')
        print('cosSim btw '+pathList[i].split('.h5')[0]+' and '+pathList[j].split('.h5')[0]+':')
        try:
            narr1=narr1[index,:]
            narr2=narr2[index,:]
        except NameError:
            narr1=narr1
            narr2=narr2
#         print(cosine_similarity(narr1, narr2),'\n')
        print(meanCosSim(narr1, narr2), '\n')
        del narr1, narr2