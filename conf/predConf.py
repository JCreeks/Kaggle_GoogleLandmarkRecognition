#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: Jing Guo
@time  : 4/25/18
"""

import os, sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)
from conf.configure import *
from utils.data_util import save_obj, load_obj
from preprocess.generator import make_generators

import time

pred_batch_size = 32 #32
pred_lr = 0.0002

pred_model = data_folder + 'xcpetion_model_.json'
pred_weights = data_folder + 'continue_second_phase_logs/older/1521093898_xcpetion_model.h5'
# pred_model_path = data_folder + '2nd_2nd_phase_xcpetion_model.h5'
pred_model_path = data_folder + 'continue_second_phase_xcpetion_model.h5'

if not os.path.exists(working_folder+'class_indices_dict'+'.pkl'):
#     classList = [f for f in os.listdir(train_class_images_path) if not os.path.isfile(os.path.join(train_class_images_path, f))]
#     classDict = dict(zip(classList, list(range(len(classList)))))
    train_img_class_gen, val_img_class_gen=make_generators(isPlain=True)
#     print(train_img_class_gen.class_indices)
    classDict = train_img_class_gen.class_indices
    save_obj(classDict, 'class_indices_dict')

class_indices_dict = load_obj('class_indices_dict')
inverted_class_indices_dict = dict((v, k) for k, v in class_indices_dict.items())
# print(type(inverted_class_indices_dict[10]))
# print(inverted_class_indices_dict[10])
