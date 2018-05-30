import pickle
# from cv2 import imread, resize
# from PIL import Image
from pathlib import Path
import random
import calendar
import time
import os, sys

from shutil import copyfile

from matplotlib.pyplot import imshow
import numpy as np
from PIL import Image

from random import shuffle

from keras.preprocessing.image import ImageDataGenerator

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)
from utils.data_util import save_obj, load_obj
from utils.siamese_util import siamese_generator
from conf.configure import *
from conf.generatorConf import *

def load_files():
    global csv_name_id_tuples_list, csv_ids_list, csv_ids_set, csv_names_set, \
    csv_id_name_dict, csv_name_id_dict, classes_num, \
    train_names_list, train_name_id_dict, val_names_list, val_name_id_dict

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
    
def add_noise(img):
    img /= 255.
    mean = 0.5
    var = 0.05
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, input_shape)
    noisy = img + gauss
    img = noisy.clip(0, 1)
    return img

def make_generators(isSimple=False, isPlain=False):
#     load_files()
    
#     global train_img_class_gen, val_img_class_gen
    folderList = [train_class_images_path, val_class_images_path
                 ]
    for folder in folderList:
        for idd in csv_ids_set:
            new_dir = folder + str(idd)
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
    
    if isSimple:
        image_class_generator = ImageDataGenerator(samplewise_center=True,
                                                          samplewise_std_normalization=True,
                                                          rescale=1/255.)
    elif isPlain:
        image_class_generator = ImageDataGenerator(rescale=1./ 255,
                                                        shear_range=0.2,
                                                        zoom_range=0.2,
                                                        horizontal_flip=True)
        
    else:
        image_class_generator = ImageDataGenerator(samplewise_center=True,
                                                   samplewise_std_normalization=True,
                                                   rotation_range=30,
                                                   width_shift_range=0.25,
                                                   height_shift_range=0.25,
                                                   zoom_range=0.3,
                                                   horizontal_flip=True,
                                                   preprocessing_function=add_noise)

    print('building image generators')
    train_img_class_gen = image_class_generator.flow_from_directory(directory=train_class_images_path,
                                                                        target_size=input_shape[:2],
                                                                        batch_size=batch_size)

    val_img_class_gen = image_class_generator.flow_from_directory(directory=val_class_images_path,
                                                                        target_size=input_shape[:2],
                                                                        batch_size=batch_size)
    print('Done building image generators')
    
    return train_img_class_gen, val_img_class_gen

def reverseDict(name_id_dict):
    id_name_dict = dict()
    for filename in name_id_dict.keys():
        idd = name_id_dict[filename]
        if idd in id_name_dict.keys():
            id_name_dict[idd].add(filename)
        else:
            id_name_dict[idd] = {filename}
    return id_name_dict

def make_siamese_generators():
    train_id_name_dict = reverseDict(train_name_id_dict)
    val_id_name_dict = reverseDict(val_name_id_dict)
    train_img_gen = siamese_generator(train_id_name_dict, train_images_folder, batch_size, img_size=input_shape)
    val_img_gen = siamese_generator(val_id_name_dict, val_images_folder, batch_size, img_size=input_shape)
    return train_img_gen, val_img_gen