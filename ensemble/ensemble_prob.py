
# coding: utf-8

# In[1]:


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


# In[2]:


os.system('rm ../output/sub*')


# In[3]:


whole_ids_list = make_ids_list(test_images_folder)
total_ids = len(whole_ids_list)


# In[4]:


xcpetion1_0 = '1525550924_xcpetion_model.h5'#.029
xcpetion1_1 = '1525597648_xcpetion_model.h5'#.059
xcpetion1_2 = '1525640444_xcpetion_model.h5'#.046
xcpetion2 = '2nd_phase_xcpetion_model.h5'#.051
xcpetion3 = '3rd_phase_xcpetion_model.h5'#.064
resnet1 = '1st_phase_resnet_model.h5'#.047
resnet3 = '3rd_phase_resnet_model.h5'#.081
inceptRes1 = '1st_phase_inceptionResnet_model.h5'#.050
inceptRes3 = '3rd_phase_inceptionResnet_model.h5'#.064
vgg3 = '3rd_phase_VGG_model.h5'#.027
dense1 = '1st_phase_denseNet_model.h5'#.083
dense3 = '3rd_phase_denseNet_model.h5'#.081
inception3 = '3rd_phase_inception_model.h5'#.081

# pathList = [xcpetion1_0,xcpetion1_1,xcpetion1_2,xcpetion2,xcpetion3]
pathList = [
            resnet3, 
            xcpetion3, 
            xcpetion2,
#             xcpetion1_1,
            inceptRes3,
#             vgg3,
            dense1,
            dense3,
            inception3,
           ]


# In[5]:


WList = [3.5,2,1,2,3.5,3.5,2]
assert len(WList)==len(pathList)


# resultPath = './result/'
# predProbList = []
# for path in pathList:
#     predProbList.append(np.load(resultPath+path+'.npy'))

# predicts = np.ones(predProbList[0].shape)
# for i, pred in enumerate(predProbList):
#     predicts*=pred**WList[i]
# predicts**=(1./sum(WList))
# certainties = np.max(predicts, axis=-1)
# labels_inds = np.argmax(predicts, axis=-1)

# predicts = np.zeros(predProbList[0].shape)
# for i, pred in enumerate(predProbList):
#     predicts+=pred*WList[i]
# predicts*=(1./sum(WList))
# certainties = np.max(predicts, axis=-1)
# labels_inds = np.argmax(predicts, axis=-1)

# In[ ]:


resultPath = './result/'

predicts = np.zeros((115743, 14951))
for i, path in enumerate(pathList):
    predicts+=((np.load(resultPath+path+'.npy'))**6)*WList[i]
    print(i)
predicts*=(1./sum(WList))
certainties = np.max(predicts, axis=-1)
labels_inds = np.argmax(predicts, axis=-1)
del predicts


# resultPath = './result/'
# 
# predicts = np.zeros((115743, 14951))
# for path in enumerate(pathList):
#     predicts=np.maximum(np.load(resultPath+path+'.npy'), predicts)
# certainties = np.max(predicts, axis=-1)
# labels_inds = np.argmax(predicts, axis=-1)
# del predicts

# In[ ]:


with open(results_file, 'w') as f:
    f.write('id,landmarks')


# In[ ]:


ids_list=whole_ids_list
with open(results_file, 'a') as f:
    text = ''
    for i in range(len(ids_list)):
        text += "\n" + str(ids_list[i]) + "," + inverted_class_indices_dict[labels_inds[i]] + " " +                str(certainties[i])
    f.write(text)


# for arr in predProbList:
#     print(arr.shape)

# In[ ]:


add_unknown_imgs(results_file)

