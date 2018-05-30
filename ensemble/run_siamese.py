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

def make_codes_by_category(train_codes_folder, codes_by_category_folder, train_name_id_dict):
    if not os.path.exists(codes_by_category_folder):
        os.makedirs(codes_by_category_folder)
        
    counter = 1
    for train_codes in listdir(train_codes_folder):
        train_codes = train_codes.replace('.pkl', '')
        print(train_codes)
        counter += 1
        temp_cats_dict = dict()
        train_codes = load_obj(train_codes, folder=train_codes_folder)
        for name, code in zip(train_codes['names'], train_codes['codes']):
            cat = train_name_id_dict[name.replace('.jpg', '')]
            if cat not in temp_cats_dict.keys():
                temp_cats_dict[cat] = dict()
                temp_cats_dict[cat]['names'] = []
                temp_cats_dict[cat]['codes'] = []
            temp_cats_dict[cat]['names'].append(name)
            temp_cats_dict[cat]['codes'].append(code)
        for cat in temp_cats_dict.keys():
            if not Path(codes_by_category_folder + str(cat)).is_file():
                cat_file = dict()
                cat_file['names'] = []
                cat_file['codes'] = []
                save_obj(obj=cat_file, name=str(cat), folder=codes_by_category_folder)
            cat_file = load_obj(str(cat), folder=codes_by_category_folder)
            names = temp_cats_dict[cat]['names']
            codes = temp_cats_dict[cat]['codes']
            cat_file['names'].extend(names)
            cat_file['codes'].extend(codes)
            save_obj(obj=cat_file, name=str(cat), folder=codes_by_category_folder)

def compute_codes(imgs_path, codes_folder, siamese_model_path):
    batch_size = 64
    code_net = get_code_net(siamese_model_path)

    if not os.path.exists(codes_folder):
        os.makedirs(codes_folder)
        
    codes_batches_size = 2**14
    print('make images names list')
    imgs_names = listdir(imgs_path)
    imgs_num = len(imgs_names)
    batches_num = imgs_num // codes_batches_size + (1 if imgs_num % codes_batches_size else 0)
    counter = 1
    this_names = []
    print('start loop')
    for ind, name in enumerate(imgs_names):
        this_names.append(name)
        # img = img_to_array(imgs_path + name)
        # imgs.append(img)
        if len(this_names) == codes_batches_size or ind == len(imgs_names) - 1:
            code_dict = {'names': this_names}
            names = this_names[:]
            steps = len(this_names) // batch_size + (1 if len(names) % batch_size else 0)
            codes = code_net.predict_generator(code_generator(names, imgs_path, batch_size=batch_size),
                                               steps=steps, verbose=2, workers=4)
            code_dict['codes'] = codes
            # codes = batch_compute_codes(code_net, imgs)
            save_obj(code_dict, 'batch_{}'.format(counter), folder=codes_folder)
            this_names = []
            print('done {} out of {}'.format(counter, batches_num))
            counter += 1
    print('codes computed and saved at: ' + codes_folder)
    
def make_categories_vectors_2(codes_folder, name_id_dict, imgs_per_cat_limit=None):
    """returns a dict with keys the categories (ids) of the images and values the 1024 size vectors for each category"""
    suf = '.jpg'
    cat_codes_dict = dict()
    counter = 1
    for codes in get_codes_from_files(codes_folder):
        for ind, name in enumerate(codes['names']):
            idd = name_id_dict[name.replace(suf, '')]
            if idd not in cat_codes_dict.keys():
                cat_codes_dict[idd] = [np.zeros(code_size), 0]
            if imgs_per_cat_limit and cat_codes_dict[idd][1] >= imgs_per_cat_limit:
                continue
            code = codes['codes'][ind] >= .5
            code = code.astype(float)
            cat_codes_dict[idd][0] += code
            cat_codes_dict[idd][1] += 1
        print('code_file', counter)
        counter += 1
    for key, value in cat_codes_dict.items():
        val = value[0] / value[1]
        val = val >= .5
        val = val.astype(float)
        cat_codes_dict[key] = val
    return cat_codes_dict

def get_results(results_file):
    """returns a dict with images names (without suffix) as keys and the result's idd/class as value"""
    names_results_ids_dict = dict()
    with open(results_file) as f:
        f.readline()
        for line in f:
            name, idd = line.strip().split(',')
            if len(idd) < 3:
                save_obj(names_results_ids_dict, 'names_results_ids_dict')
                return names_results_ids_dict
            idd = idd.split(' ')[0]
            names_results_ids_dict[name] = idd
    return names_results_ids_dict

def calc_for_each_sim(code, cat):
    cat_codes = load_obj(cat, folder=codes_by_category_folder)
    total = len(cat_codes['codes'])
    similarity = 0
    for cat_code in cat_codes['codes']:
        similarity += 1 - cosine(code, cat_code)
    return similarity / total

def compute_similarities_for_each(test_codes_folder, old_results_file):
    suf = '.jpg'
    test_img_similarity_dict = dict()
    names_results_ids_dict = get_results(old_results_file)
    counter = 1
    for codes_dict in get_codes_from_files(test_codes_folder):
        for name, code in zip(codes_dict['names'], codes_dict['codes']):
            name = name.replace(suf, '')
            cat = names_results_ids_dict[name]
            similarity = calc_for_each_sim(code, cat)
            test_img_similarity_dict[name] = similarity
            # min_distance = distance if distance < min_distance else min_distance
        print(counter)
        counter += 1
    # test_img_similarity_dict = distance_to_similarity(test_img_similarity_dict, min_distance)
    save_obj(test_img_similarity_dict, 'test_img_similarity_dict'+old_results_file.split('/')[-1])
    return test_img_similarity_dict

def change_results_from_code_files(test_img_similarity_dict, old_results_file, new_results_file, old_prop_rate=0.5):
    with open(old_results_file) as fin:
        with open(new_results_file, 'w') as fout:
            fout.write(fin.readline())
            for line in fin:
                name, idd = line.strip().split(',')
                if len(idd.split(' ')) == 2:
                    old_prob = idd.split(' ')[1]
                    sim_prob = str(test_img_similarity_dict[name])
                    new_prob = calc_new_prob(old_prob, sim_prob, old_prop_rate)
                    line = line.replace(old_prob, str(new_prob))
                fout.write(line)

def run_tests(old_prop_rate = 0, new_results_file=new_results_file):
#     cat_codes_dict = load_obj('cat_codes_dict', folder=working_folder)
    try:
        img_similarity_dict = load_obj('test_img_similarity_dict'+old_results_file.split('/')[-1], folder=working_folder)
    except:
        img_similarity_dict = compute_similarities_for_each(test_codes_folder, old_results_file)
    
    change_results_from_code_files(img_similarity_dict, old_results_file, new_results_file, old_prop_rate)
    
if __name__ == '__main__':
    print('start excecution')

    # compute_codes(train_images_folder, train_codes_folder, '../siamese/'+'1st_phase_siamese_model.h5')
    # make_codes_by_category(train_codes_folder, codes_by_category_folder, train_name_id_dict)

    # compute_codes(val_images_folder, val_codes_folder, '../siamese/'+'1st_phase_siamese_model.h5')
    # make_codes_by_category(val_images_folder, codes_by_category_folder, val_name_id_dict)

    # compute_codes(test_images_folder, test_codes_folder, '../siamese/'+'1st_phase_siamese_model.h5')

    # try:
    #     cat_codes_dict = load_obj('cat_codes_dict', folder=working_folder)
    # except:
    #     cat_codes_dict = make_categories_vectors_2(test_codes_folder, val_name_id_dict)
    #     save_obj(cat_codes_dict, name='cat_codes_dict', folder=working_folder)

    run_tests(old_prop_rate = 0, new_results_file=new_results_file)
