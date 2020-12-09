from __future__ import print_function
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dropout, Flatten, Dense, LSTM, TimeDistributed, Bidirectional
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import multiply, Permute, Reshape
from tensorflow.keras.models import load_model
import tensorflow.keras.optimizers as optimizers
from sklearn import metrics
import numpy as np
#import tensorflow.config as config
from tensorflow.keras.utils import multi_gpu_model
from sklearn.metrics import confusion_matrix
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn import manifold


h5_dir='/home/lab210/MattYen_workspace/gest_August/cross-scene/3t4rRDI/'


x_train = np.load(h5_dir + 'x_train.npy')
y_train = np.load(h5_dir + 'y_train.npy') 

#x_test_blank = np.load(blank_dir + 'x_blank.npy')

x_test = np.load(h5_dir + 'x_test.npy')
y_test = np.load(h5_dir + 'y_test.npy') 
#print (x_test.shape)
#print (y_test.shape) 

x_train = np.transpose(x_train, (0,1,3,4,2))
x_test = np.transpose(x_test, (0,1,3,4,2))

x_train = x_train[:,:,:,:,0]
x_train = x_train[..., np.newaxis]
x_test = x_test[:,:,:,:,0]
x_test = x_test[..., np.newaxis]

#============================================================
x_train_new = np.load(h5_dir + 'x_train.npy')
y_train_new = np.load(h5_dir + 'y_train.npy') 

#x_test_blank = np.load(blank_dir + 'x_blank.npy')

x_test_new = np.load(h5_dir + 'x_test.npy')
y_test_new = np.load(h5_dir + 'y_test.npy') 
#print (x_test.shape)
#print (y_test.shape) 

x_train_new = np.transpose(x_train_new, (0,1,3,4,2))
x_test_new = np.transpose(x_test_new, (0,1,3,4,2))

x_train_new = x_train_new[:,:,:,:,0]
x_train_new = x_train_new[..., np.newaxis]
x_test_new = x_test_new[:,:,:,:,0]
x_test_new = x_test_new[..., np.newaxis]
#============================================================
#90-10
temp_train_x = np.zeros((1440,64,32,32,1))
temp_train_y = np.zeros((1440,64,12))
temp_test_x = np.zeros((1440,64,32,32,1))
temp_test_y = np.zeros((1440,64,12))

for i in range(0, 1440):
    if i % 10 == 9:
        temp_train_x[i,...] = x_train[i,...]
        temp_train_y[i,...] = y_train[i,...]
        
for i in range(0, 1440):
    if i % 10 == 9:
        temp_test_x[i,...] = x_test[i,...]
        temp_test_y[i,...] = y_test[i,...]

for i in range(0, 1440):
    if i % 10 == 9:
        x_train_new[i,...] = temp_test_x[i,...]
        y_train_new[i,...] = temp_test_y[i,...]
        
for i in range(0, 1440):
    if i % 10 == 9:
        x_test_new[i,...] = temp_train_x[i,...]
        y_test_new[i,...] = temp_train_y[i,...]
        
del x_train, y_train, x_test, y_test
del temp_train_x, temp_train_y, temp_test_x, temp_test_y