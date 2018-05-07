#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: Jing Guo
@time  : 4/25/18
"""
import os, sys
from PIL import Image

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)
from conf.configure import *
from utils.data_util import *
from conf.predConf import *
from conf.generatorConf import *

import numpy as np, pandas as pd
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
from sklearn.metrics import roc_auc_score

def add_unknown_imgs(results_file):
    test_csv_names_list = make_csv_list(test_csv_path)
    csv_names_set = set(test_csv_names_list)
    with open(results_file, 'r') as f:
            f.readline()
            for line in f:
                l = line.strip().split(',')
                if len(l) != 2:
                    print(l)
                    continue
                name = l[0]
                csv_names_set.remove(name)

    with open(results_file, 'a') as f:
            for name in csv_names_set:
                line = '\n' + name + ','
                f.write(line)


def pred_generator(ids_list, imgs_dir, batch_size, img_size=(224, 224, 3), suffix='.jpg', normalize=True):
    while len(ids_list) > 0:
        batch_size = pred_batch_size if len(ids_list) >= pred_batch_size else len(ids_list)
        imgs = np.zeros((batch_size, *img_size))
        for i in range(batch_size):
            name = ids_list.pop(0)
            img = Image.open(imgs_dir + name + suffix)
            img = np.asarray(img.resize(img_size[:2], Image.ANTIALIAS))
            img = img / 255.0

            if normalize:
                # mean
                img_mean = np.mean(img, axis=(0, 1))
                img = img - img_mean

                #std
                img_std = np.std(img, axis=(0, 1))
                img = img / img_std

            imgs[i] = img
        yield imgs

def split_seq(seq, size):
    """ Split up seq in pieces of size """
    return [seq[i:i+size] for i in range(0, len(seq), size)]

def gap(y_true, y_pred):
    arg_true = y_true
    arg_pred = y_pred
    val_pred = K.max(arg_pred, axis=-1)
    _, sorted_indices = tf.nn.top_k(val_pred, batch_size)

    new_arg_pred = []
    new_arg_true = []

    for i in range(batch_size):
        _pred = arg_pred[sorted_indices[i]]
        _true = arg_true[sorted_indices[i]]
        new_arg_pred.append(_pred)
        new_arg_true.append(_true)

    new_arg_pred = K.stack(new_arg_pred)
    new_arg_true = K.stack(new_arg_true)

    arg_pred = new_arg_pred
    arg_true = new_arg_true

    arg_pred = K.argmax(arg_pred, axis=-1)
    arg_true = K.argmax(arg_true, axis=-1)

    correct_pred = K.equal(arg_pred, arg_true)
    precision = [K.switch(K.gather(correct_pred, i - 1),
                          K.sum(K.cast(correct_pred, 'float32')[:i]) / i,
                          K.variable(0))
                 for i in range(1, batch_size + 1)]
    precision = K.stack(precision)
    _gap = K.sum(precision) / K.sum(K.cast(correct_pred, 'float32'))
    return _gap


def square_error(y_true, y_pred):
    return K.sum(K.square(y_pred - y_true), axis=-1)

class GapCallback(keras.callbacks.Callback):
    def __init__(self, validation_generator, validation_steps):
        self.validation_generator = validation_generator
        self.validation_steps = validation_steps

    def on_train_begin(self, logs={}):
        self.preds = []
        self.trues = []
        self.probs = []

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        for batch_index in range(self.validation_steps):
            features, y_true = next(self.validation_generator)
            y_pred = np.asarray(self.model.predict(features))
            print(y_pred.shape)

            y_true = np.argmax(y_true, -1)
            prob = np.max(y_pred, -1)
            y_pred = np.argmax(y_pred, -1)
            print(y_pred.shape)

            self.preds.extend(y_pred.tolist())
            self.trues.extend(y_true.tolist())
            self.probs.extend(prob.tolist())
            print(y_pred.shape)

            y_preds = np.array(self.preds)
            y_trues = np.array(self.trues)
            probs = np.array(self.probs)

            # y_preds = np.reshape(y_preds, (-1, y_preds.shape[-1]))
            # y_trues = np.reshape(y_trues, (-1, y_trues.shape[-1]))
            # probs = np.reshape(probs, (-1, probs.shape[-1]))
            print(y_true.shape, y_pred.shape, y_trues.shape, y_preds.shape)

            true_pos = [true_label == pred_label for true_label, pred_label in zip(y_trues, y_preds)]
            # print(true_pos)
            true_pos = [x for _, x in sorted(zip(probs, true_pos), reverse=True)]
            # print(true_pos)
            gap = 0
            for i in range(len(true_pos)):
                precision = sum(true_pos[:i + 1]) / len(true_pos[:i + 1])
                gap += precision if true_pos[i] else 0
            gap /= len(true_pos)
            print(gap)

            # _gap = label_ranking_average_precision_score(y_true=y_trues, y_score=y_preds)
            # print(_gap)
            return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

def set_reg_drop(model, reg_num, drop_rate):
    for layer in model.layers:
        if isinstance(layer, keras.layers.Dropout):
            model.get_layer(layer.name).rate = drop_rate
        if not layer.trainable:
            continue
        if isinstance(layer, (keras.layers.convolutional.Conv2D, keras.layers.Dense)):
            model.get_layer(layer.name).kernel_regularizer = regularizers.l2(reg_num)
            model.get_layer(layer.name).activity_regularizer = regularizers.l1(reg_num)
    return model

class RocAucMetricCallback(keras.callbacks.Callback):
    def __init__(self, predict_batch_size=batch_size, include_on_batch=False):
        super(RocAucMetricCallback, self).__init__()
        self.predict_batch_size=predict_batch_size
        self.include_on_batch=include_on_batch

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        if(self.include_on_batch):
            logs['roc_auc_val']=float('-inf')
            if(self.validation_data):
                logs['roc_auc_val']=roc_auc_score(self.validation_data[1], 
                                                  self.model.predict(self.validation_data[0],
                                                                     batch_size=self.predict_batch_size))

    def on_train_begin(self, logs={}):
        if not ('roc_auc_val' in self.params['metrics']):
            self.params['metrics'].append('roc_auc_val')

    def on_train_end(self, logs={}):
        pass

    def on_epoch_begin(self, epoch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        logs['roc_auc_val']=float('-inf')
        if(self.validation_data):
            logs['roc_auc_val']=roc_auc_score(self.validation_data[1], 
                                              self.model.predict(self.validation_data[0],
                                                                 batch_size=self.predict_batch_size))
            
def GAP_vector(pred, conf, true, return_x=False):
    '''
    Compute Global Average Precision (aka micro AP), the metric for the
    Google Landmark Recognition competition. 
    This function takes predictions, labels and confidence scores as vectors.
    In both predictions and ground-truth, use None/np.nan for "no label".

    Args:
        pred: vector of integer-coded predictions
        conf: vector of probability or confidence scores for pred
        true: vector of integer-coded labels for ground truth
        return_x: also return the data frame used in the calculation

    Returns:
        GAP score
    '''
    x = pd.DataFrame({'pred': pred, 'conf': conf, 'true': true})
    x.sort_values('conf', ascending=False, inplace=True, na_position='last')
    x['correct'] = (x.true == x.pred).astype(int)
    x['prec_k'] = x.correct.cumsum() / (np.arange(len(x)) + 1)
    x['term'] = x.prec_k * x.correct
    gap = x.term.sum() / x.true.count()
    if return_x:
        return gap, x
    else:
        return gap