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

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)
from utils.data_util import save_obj, load_obj
from conf.configure import *

csv_name_id_tuples_list = []
csv_name_id_dict = dict()
csv_id_name_dict = dict()
csv_names_list = []
csv_ids_list = []

csv_name_id_tuples_list = []
csv_ids_list = []
csv_ids_set = set()
csv_names_set = set()

csv_id_name_dict = dict()
csv_name_id_dict = dict()
train_names_list = dict()
train_name_id_dict = dict()
classes_num = 0

def make_folder_lists_dicts(folder_path, suffix=r'.jpg'):
    names_list = []
    name_id_dict = dict()
    for filename in os.listdir(folder_path):
        filename = filename.replace(suffix, '')
        if filename not in csv_names_set:
            print(filename)
            continue
        names_list.append(filename)
        name_id_dict[filename] = csv_name_id_dict[filename]
    return names_list, name_id_dict

def make_files():
    global csv_name_id_tuples_list, csv_ids_list, csv_ids_set, csv_names_set, \
    csv_id_name_dict, csv_name_id_dict, classes_num, \
    train_names_list, train_name_id_dict, val_names_list, val_name_id_dict
    
    print("parsing train.csv")
    with open(csv_csv_path) as f:
        f.readline()
        for line in f:
            l = line.replace('"', '').strip().split(',')
            if len(l) != 3:
                print(l)
                continue
            name, idd = l[0], int(l[2])
            csv_name_id_tuples_list.append((name, idd))
            csv_names_list.append(name)

            csv_name_id_dict[name] = idd
            csv_ids_list.append(idd)

            if idd in csv_id_name_dict.keys():
                csv_id_name_dict[id].add(name)
            else:
                csv_id_name_dict[id] = {name}
    
    print("start saving lists")
    csv_names_set = set(csv_names_list)
    csv_ids_set = set(csv_ids_list)

    save_obj(csv_name_id_tuples_list, 'csv_name_id_tuples_list')
    save_obj(csv_names_list, 'csv_names_list')
    save_obj(csv_ids_list, 'csv_ids_list')

    save_obj(csv_name_id_dict, 'csv_name_id_dict')
    save_obj(csv_id_name_dict, 'csv_id_name_dict')

    save_obj(csv_ids_set, 'csv_ids_set')
    save_obj(csv_names_set, 'csv_names_set')

    train_names_list, train_name_id_dict = make_folder_lists_dicts(train_images_folder)
    val_names_list, val_name_id_dict = make_folder_lists_dicts(val_images_folder)

    save_obj(train_names_list, 'train_names_list')
    save_obj(train_name_id_dict, 'train_name_id_dict')

    save_obj(val_names_list, 'val_names_list')
    save_obj(val_name_id_dict, 'val_name_id_dict')

if __name__ == '__main__':
    folderList = [working_folder, train_images_folder, val_images_folder, test_images_folder, new_images_folder, data_folder, csv_csv_path]
    for folder in folderList:
        if not os.path.exists(folder):
                os.makedirs(folder)
                
    make_files()
    print('done making files')

    print('done')
