#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: Jing Guo
@time  : 4/25/18
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

# import cPickle
import pickle
import pandas as pd
from conf.configure import *

def save_obj(obj, name, folder=working_folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    with open(folder + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name, folder=working_folder):
    with open(folder + name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
def make_ids_list(folder_path, suffix='.jpg'):
    names_list = []
    for filename in os.listdir(folder_path):
        filename = filename.replace(suffix, '')
        names_list.append(filename)
    names_list.sort()
    return names_list

def load_model_from_file(path):
    # load json and create model
    with open(path, 'r') as json_file:
        model = model_from_json(json_file.read())
    return mode


def make_csv_list(csv_path):
    names_list = []
    with open(csv_path) as f:
        f.readline()
        for line in f:
            l = line.replace('"', '').strip().split(',')
            if len(l) != 2:
                print(l)
                continue
            name = l[0]
            names_list.append(name)
    return names_list

