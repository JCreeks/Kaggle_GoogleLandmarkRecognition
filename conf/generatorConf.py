#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: Jing Guo
@time  : 4/25/18
"""

import os, sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)
from utils.data_util import save_obj, load_obj
from conf.configure import *

#input data
batch_size = 32
input_shape = (224, 224, 3)
img_size = input_shape

#list dict
csv_name_id_tuples_list = load_obj('csv_name_id_tuples_list')
csv_ids_list = load_obj('csv_ids_list')
csv_ids_set = load_obj('csv_ids_set')
csv_names_set = load_obj('csv_names_set')

csv_id_name_dict = load_obj('csv_id_name_dict')
csv_name_id_dict = load_obj('csv_name_id_dict')

train_names_list = load_obj('train_names_list')
train_name_id_dict = load_obj('train_name_id_dict')

val_names_list = load_obj('val_names_list')
val_name_id_dict = load_obj('val_name_id_dict')
classes_num = len(csv_ids_set)
#train_classes_num = len(os.listdir(train_class_images_path))

#xception 
first_phase_epochs = 1
second_phase_epochs = 1
third_phase_epochs = 1
cnn_first_phase_epochs = 1
dense_phase_epochs =1

saves_per_epoch = 10
small_epochs = 50

imgs_per_rep = int(len(train_names_list) / saves_per_epoch)
imgs_per_small_epoch = int(imgs_per_rep / small_epochs)
steps_per_small_epoch = int(imgs_per_small_epoch / batch_size)

first_phase_train_reps = first_phase_epochs * saves_per_epoch
second_phase_train_reps = second_phase_epochs * saves_per_epoch
third_phase_train_reps = third_phase_epochs * saves_per_epoch
cnn_first_phase_train_reps = cnn_first_phase_epochs * saves_per_epoch
dense_phase_train_reps = dense_phase_epochs * saves_per_epoch

val_size = len(val_names_list)
val_imgs_per_rep = int(val_size / saves_per_epoch)
val_imgs_per_small_epoch = int(val_imgs_per_rep / small_epochs)
val_steps_per_small_epoch = int(val_imgs_per_small_epoch / batch_size) * 10

