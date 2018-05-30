
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

import time

# working_folder = r'/home/ubuntu/KaggleCompetition/LandmarkRecognition/input/'
working_folder = r'../input/'
train_images_folder = working_folder + r'train_images/'
val_images_folder = working_folder + r'val_images/'
test_images_folder = working_folder + r'testImage/'
new_images_folder = working_folder + r'trainImage/'
# new_images_folder = working_folder + r'new_toy_images/'
train_class_images_path = working_folder + 'train_class_images/'
val_class_images_path = working_folder + 'val_class_images/'

# data_folder = working_folder + r''
csv_csv_path = working_folder + 'train.csv'
test_csv_path = working_folder + 'test.csv'

#xception trained models
data_folder = r'../nasnet/'#r'../xception/'
first_phase_folder = data_folder + r'first_phase_logs/'
second_phase_folder = data_folder + r'second_phase_logs/'
second_second_phase_folder = data_folder + r'second_second_phase_logs/'
continue_second_phase_folder = data_folder + r'continue_second_phase_logs/'
third_phase_folder = data_folder + r'third_phase_logs/'
smoothed_third_phase_folder = data_folder + r'smoothed_third_phase_logs/'
double_drop_hinge_phase_folder = data_folder + 'double_drop_hinge_phase_logs/'
drop_ase_xcpetion_phase_folder = data_folder + 'drop_ase_phase_logs/'

cnn_folder = r'../cnn/'
cnn_first_phase_folder = cnn_folder + r'first_phase_logs/'

dense_folder = r'../dense/'
dense_phase_folder = dense_folder + r'dense_logs/'

this_model = data_folder + "xcpetion_model_dropout_2048_1024.json"
this_model_weights = data_folder + 'continue_second_phase_logs/older/1520948479_xcpetion_model.h5'

second_second_phase_model = second_second_phase_folder + '1520424331_xcpetion_model.h5'
double_drop_hinge_phase_model = double_drop_hinge_phase_folder + '1520855530_xcpetion_model.h5'
drop_ase_xcpetion_model_path = drop_ase_xcpetion_phase_folder + '1520589478_xcpetion_model.h5'
initial_model = smoothed_third_phase_folder + '_xcpetion_model.h5'

train_codes_folder = r'../siamese/' + 'train_codes/'
val_codes_folder = r'../siamese/' + 'val_codes/'
test_codes_folder = r'../siamese/' + 'test_codes/'
codes_by_category_folder = r'../siamese/' + 'codes_by_category/'
code_size = 1024

results_file = '../output/submission_{}.csv'.format(time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time())))
old_results_file = '../output/Pt114.csv'
new_results_file = old_results_file.split('.csv')[0]+'_ensSiamese.csv'

topK = 5
old_results_file_topK = '../output/0529_2'+'top'+str(topK)+'.csv'
new_results_file_topK = old_results_file_topK.split('.csv')[0]+'_ensSiamese.csv'


    
