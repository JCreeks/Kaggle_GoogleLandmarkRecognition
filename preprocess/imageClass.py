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

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)
from utils.data_util import save_obj, load_obj
from conf.configure import *

folderList = [working_folder, train_images_folder, val_images_folder, test_images_folder, new_images_folder, data_folder, csv_csv_path]
for folder in folderList:
    if not os.path.exists(folder):
            os.makedirs(folder)

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

train_class_images_path = working_folder + 'train_class_images/'
val_class_images_path = working_folder + 'val_class_images/'

# def change_size(imgs_folder, target_size=(224, 224, 3)):
#     size = (target_size[0], target_size[1])
#     total_imgs = len(list(os.listdir(imgs_folder)))
#     counter = 0
#     for filename in os.listdir(imgs_folder):
#         try:
#             img = Image.open(imgs_folder + filename)
#         except OSError:
#             print(filename)
#             continue
#         img = img.resize(size, Image.ANTIALIAS)
#         img.save(imgs_folder + filename)
#         if counter % 100 == 0:
#             print(counter, ' out of ', total_imgs)
#         counter += 1


# def make_class_images(folder_path, classes_path, suffix=r'.jpg'):
#     for filename in os.listdir(folder_path):
#         idd = csv_name_id_dict[filename.replace(suffix, '')]
#         new_dir = classes_path + str(idd)
#         if not os.path.exists(new_dir):
#             os.makedirs(new_dir)
#         copyfile(folder_path + filename, new_dir + r'//' + filename)


def make_class_images_ratio(folder_path, category_1_path, classes_1_path, category_2_path, classes_2_path, sampleRatio=.5, trainRatio=0.5, suffix=r'.jpg'):
    folderList = [category_1_path, category_2_path, classes_1_path, classes_2_path
                 ]
    for folder in folderList:
        os.system('rm -r '+ folder)
        if not os.path.exists(folder):
                os.makedirs(folder)
    
    imageList = os.listdir(folder_path)
    shuffle(imageList)
    imageList = imageList[:int(len(imageList)*sampleRatio)]
    total_imgs = len(imageList)
    classes_1_num = int(total_imgs * trainRatio)
    
    index = 0
    for filename in imageList:
        (category_path, classes_path) = (category_1_path, classes_1_path) if index < classes_1_num else (category_2_path, classes_2_path)

        idd = csv_name_id_dict[filename.replace(suffix, '')]
        new_dir = classes_path + str(idd)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        copyfile(folder_path + filename, category_path + filename)
        copyfile(folder_path + filename, new_dir + r'/' + filename)
        # print(filename, idd)
        if index % 1000 == 0:
            print(index, ' out of ', total_imgs)
        index += 1

if __name__ == '__main__':
    # change_size(new_images_folder)
    # make_class_images(val_images_folder, val_class_images_path)
    # make_class_images(train_images_folder, train_class_images_path)
    
    folderList = [train_class_images_path, val_class_images_path
                 ]
    for folder in folderList:
        if not os.path.exists(folder):
                os.makedirs(folder)
                
    sampleRatio = 1.
    trainRatio = .9
    print("sampleRatio", sampleRatio)
    print("train ratio", trainRatio)
   
    make_class_images_ratio(new_images_folder, train_images_folder, train_class_images_path,
                            val_images_folder, val_class_images_path, sampleRatio=sampleRatio,
                            trainRatio=trainRatio)

    print('done')
