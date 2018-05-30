import pickle
# from cv2 import imread, resize
# from PIL import Image
from pathlib import Path
import random
import calendar
import time
import os, sys
from  os import listdir

import warnings
warnings.filterwarnings('ignore')

from matplotlib.pyplot import imshow
import numpy as np
from PIL import Image

import numpy as np
from scipy.spatial.distance import cosine, euclidean, hamming
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
from utils.siamese_util import set_trainables, code_generator, get_code_net, calc_new_prob
from conf.configure import *
from conf.generatorConf import *
from preprocess.generator import make_siamese_generators

def get_codes_from_files(codes_folder):
    """ yields the codes (dict) from each file in the codes folder"""
    for filename in listdir(codes_folder):
        filename = filename.replace('.pkl', '')
        codes = load_obj(filename, folder=codes_folder)
        yield codes

def get_results_topK(old_results_file_topK, k=topK):
    """returns a dict with images names (without suffix) as keys and the result's idd/class as value"""
    names_results_ids_dict = dict()
    with open(old_results_file_topK) as f:
        f.readline()
        for line in f:
            name, idd = line.strip().split(',')
            if len(idd) < 3:
                save_obj(names_results_ids_dict, 'names_results_ids_dict_topK')
                return names_results_ids_dict
            names_results_ids_dict[name] = {}
            iddList = idd.split(' ')
            for i in range(k):
                names_results_ids_dict[name][iddList[2*(topK-1-i)]] = float(iddList[2*(topK-1-i)+1])
    return names_results_ids_dict

def calc_for_each_sim(code, cat):
    cat_codes = load_obj(cat, folder=codes_by_category_folder)
    total = len(cat_codes['codes'])
    similarity = 0
    for cat_code in cat_codes['codes']:
        similarity += 1 - cosine(code, cat_code)
    return similarity / total

def ensFunc(prob, sim):
    return prob**(-np.log(sim))
#    return prob**(1./sim-1)
#    return prob**(1-sim)
#    return prob**(np.tan((1-sim)*np.pi/2))
#    return sim**(1-prob)

def compute_similarities_for_each_topK(test_codes_folder, old_results_file_topK, topK=topK):
    suf = '.jpg'
    test_img_similarity_dict = dict()
    names_results_ids_dict = get_results_topK(old_results_file_topK, k=topK)
    counter = 1
    for codes_dict in get_codes_from_files(test_codes_folder):
        for name, code in zip(codes_dict['names'], codes_dict['codes']):
            name = name.replace(suf, '')
            catProbDict = names_results_ids_dict[name]
            similarity = 0
            for cat, prob in catProbDict.items():
                tmpSim = ensFunc(prob, calc_for_each_sim(code, cat))
                if tmpSim > similarity:
                    idd = cat
                    similarity = tmpSim
            test_img_similarity_dict[name] = [idd, similarity]
            # min_distance = distance if distance < min_distance else min_distance
        print(counter)
        counter += 1
    # test_img_similarity_dict = distance_to_similarity(test_img_similarity_dict, min_distance)
    save_obj(test_img_similarity_dict, 'test_sim_dict'+'_top'+str(topK)+'_'+old_results_file_topK.split('/')[-1])
    return test_img_similarity_dict

def change_results_from_code_files_topK(test_img_similarity_dict, old_results_file_topK, new_results_file_topK):
    with open(old_results_file_topK) as fin:
        with open(new_results_file_topK, 'w') as fout:
            fout.write(fin.readline())
            for line in fin:
                name, idd = line.strip().split(',')
                if len(idd.split(' ')) >= 2:
#                     old_prob = idd.split(' ')[1]
#                     sim_prob = str(test_img_similarity_dict[name])
#                     new_prob = calc_new_prob(old_prob, sim_prob, old_prop_rate)
#                     line = line.replace(old_prob, str(new_prob))
                    line = name + ',' + str(test_img_similarity_dict[name][0]) + ' ' + str(test_img_similarity_dict[name][1]) + "\n"
                fout.write(line)

def run_tests(new_results_file_topK=new_results_file_topK, topK=topK):
#     cat_codes_dict = load_obj('cat_codes_dict', folder=working_folder)
#    try:
#        img_similarity_dict = load_obj('test_sim_dict'+'_top'+str(topK)+'_'+old_results_file_topK.split('/')[-1], folder=working_folder)
#    except:
    img_similarity_dict = compute_similarities_for_each_topK(test_codes_folder, old_results_file_topK, topK=topK)
    
    change_results_from_code_files_topK(img_similarity_dict, old_results_file_topK, new_results_file_topK)
    
def compute_probabilities_for_each_topK(test_codes_folder, old_results_file_topK, k=topK):
    suf = '.jpg'
    test_img_similarity_dict = dict()
    names_results_ids_dict = get_results_topK(old_results_file_topK, k=topK)
    counter = 1
    for codes_dict in get_codes_from_files(test_codes_folder):
        for name in codes_dict['names']:
            name = name.replace(suf, '')
            catProbDict = names_results_ids_dict[name]
            similarity = 0
            for cat, prob in catProbDict.items():
                tmpSim = prob
                if tmpSim > similarity:
                    idd = cat
                    similarity = tmpSim
            test_img_similarity_dict[name] = [idd, similarity]
            # min_distance = distance if distance < min_distance else min_distance
        print(counter)
        counter += 1
    # test_img_similarity_dict = distance_to_similarity(test_img_similarity_dict, min_distance)
#     save_obj(test_img_similarity_dict, 'test_sim_dict'+'_top'+str(topK)+'_'+old_results_file_topK.split('/')[-1])
    return test_img_similarity_dict

def run_tests_probs(new_results_file_topK=results_file, k=topK):
    img_similarity_dict = compute_probabilities_for_each_topK(test_codes_folder, old_results_file_topK, k=topK)
    
    change_results_from_code_files_topK(img_similarity_dict, old_results_file_topK, new_results_file_topK)
    
if __name__ == '__main__':
    print('start excecution')

    k = 2
    run_tests(new_results_file_topK=new_results_file_topK.split('.csv')[0]+'_top'+str(k)+''+'.csv', topK=k)
    
    run_tests_probs(new_results_file_topK=results_file, k=1)
