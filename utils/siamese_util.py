# coding: utf-8

import pickle
from cv2 import imread, resize
from PIL import Image
from numbers import Number
import os, sys
from  os import listdir
from pathlib import Path
import multiprocessing
from scipy.spatial.distance import cosine, euclidean, hamming
from random import sample

import numpy as np
from keras.layers import Input, Conv2D, Dropout, merge, Dense, Flatten, MaxPooling2D, GlobalAveragePooling2D
from keras import layers
from keras.models import Model, Sequential
from keras.models import load_model, model_from_json
import keras
from keras import backend as K
import tensorflow as tf
import pandas as pd

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

def add_noise(img):
    mean = 0.5
    var = 0.05
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, img.shape)
    noisy = img + gauss
    img = noisy.clip(0, 1)
    return img

def siamese_generator(id_name_dict, imgs_dir, batch_size, img_size=(224, 224, 3), suffix='.jpg'):
    while True:
        imgs_0 = np.zeros((batch_size, *img_size))
        imgs_1 = np.zeros((batch_size, *img_size))

        targets = np.zeros((batch_size, 1))
        targets[:batch_size // 2, 0] = 1
        idds = sample(id_name_dict.keys(), batch_size)
        # idds = rng.choice(list(id_name_dict.keys()), size=batch_size, replace=False)

        for i in range(batch_size):
            idd = idds[i]
            same = False
            if i < batch_size // 2:
                try:
                    name_pair = sample(id_name_dict[idd], 2)
                    same = False
                except ValueError:
                    name = list(id_name_dict[idd])[0]
                    name_pair = [name, name]
                    same = True
            else:
                id_1 = idds[i]
                name_1 = sample(id_name_dict[id_1], 1)[0]

                id_2 = sample(set(id_name_dict.keys()).difference({id_1}), 1)[0]
                name_2 = sample(id_name_dict[id_2], 1)[0]
                name_pair = [name_1, name_2]

            img_pair = [Image.open(imgs_dir + name + suffix) for name in name_pair]
            img_pair = [np.asarray(img.resize(img_size[:2], Image.ANTIALIAS)) for img in img_pair]
            img_pair = [img / 255.0 for img in img_pair]

            # mean
            img_mean_pair = [np.mean(img, axis=(0, 1)) for img in img_pair]
            img_pair = [img - img_mean for img, img_mean in zip(img_pair, img_mean_pair)]

            #std
            img_std_pair = [np.std(img, axis=(0, 1)) for img in img_pair]
            img_pair = [img / img_std for img, img_std in zip(img_pair, img_std_pair)]
            imgs_0[i] = img_pair[0]
            imgs_1[i] = img_pair[1] if not same else add_noise(img_pair[1])
        yield [imgs_0, imgs_1], targets

def set_trainables(model, choice):
    if isinstance(choice, Number):
        ratio = choice
        trainable_layers_index = int(len(model.layers) * (1 - ratio))
        for layer in model.layers[:trainable_layers_index]:
            layer.trainable = False
        for layer in model.layers[trainable_layers_index:]:
            layer.trainable = True
    else:
        for layer in model.layers:
            layer.trainable = layer.name in choice
    return model

def load_model_from_file(path):
    # load json and create model
    with open(path, 'r') as json_file:
        model = model_from_json(json_file.read())
    return mode

def load_weights_from_model(model, old_model):
    for layer in model.layers:
        if isinstance(layer, (keras.layers.Dropout, keras.layers.InputLayer)):
            continue
#         print(layer.name)
        weights = old_model.get_layer(layer.name).get_weights()
        model.get_layer(layer.name).set_weights(weights)
    return model

def get_code_net(siamese_model_path, model_path='../resnet/' + '3rd_phase_resnet_model.h5'):
#     siamese_net = SiameseModel(data_folder + '2nd_phase_xcpetion_model.h5')
#     siamese_net.load_weights(siamese_weights_path)
    siamese_net = load_model(siamese_model_path)
    short_model = siamese_net.get_layer('model_1')

#     xcpetion_model = load_model_from_file(data_folder + 'xcpetion_model_.json')
    xcpetion_model = load_model(model_path)
    xcpetion_model.get_layer('dense_1').activation = K.sigmoid
    code_net = Model(inputs=xcpetion_model.input, outputs=xcpetion_model.get_layer('dense_1').output)

    code_net = load_weights_from_model(code_net, short_model)
    return code_net
        
def code_generator(names, imgs_dir, batch_size=64, img_size=(224, 224, 3)):
    while len(names) > 0:
        batch_size = batch_size if len(names) >= batch_size else len(names)
        imgs = np.zeros((batch_size, *img_size))
        for i in range(batch_size):
            name = names.pop(0)
            img = Image.open(imgs_dir + name)
            img = np.asarray(img.resize(img_size[:2], Image.ANTIALIAS))
            img = img / 255.0

            # mean
            img_mean = np.mean(img, axis=(0, 1))
            img = img - img_mean

            #std
            img_std = np.std(img, axis=(0, 1))
            img = img / img_std

            imgs[i] = img
        yield imgs

def calc_new_prob(old_prob, sim_prob, old_prop_rate=0.5):
    old_prob, sim_prob = float(old_prob), float(sim_prob)
    # if old_prob > .9:
    #     return old_prob
    # if sim_prob > .9:
    #     return sim_prob
    new_prob = (old_prob * old_prop_rate) + (sim_prob * (1 - old_prop_rate))
    return new_prob
