

from __future__ import print_function
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dropout, Flatten, Dense, LSTM, TimeDistributed, Bidirectional
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import multiply, Permute, Reshape, Input, concatenate
from tensorflow.keras.models import load_model, Model
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




#tf.debugging.set_log_device_placement(True)
#gpus = tf.config.experimental.list_logical_devices('GPU')
#if gpus:
#    c=[]
#    for gpu in gpus:
#        with tf.device(gpu.name):
#            a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
#            b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
#            c.append(tf.matmul(a, b))
#        with tf.device('/CPU:0'):
#            print(tf.add_n(c))


h5_dir_1='/home/lab210/MattYen_workspace/gest_August/cross-scene/3t4rRDI/'
h5_dir_2='/home/lab210/MattYen_workspace/gest_August/cross-scene/1t2rRAI/'
#model_dir = '/home/pstudent/DeepSoli Models/gesture_data/[parallel model]/1t2r_parallel/'
vali_dir_1='/home/lab210/MattYen_workspace/gest_August/cross-scene/3t4rRDI/'
vali_dir_2='/home/lab210/MattYen_workspace/gest_August/cross-scene/1t2rRAI/'

#blank_dir='/home/pstudent/DeepSoli Models/gesture_data/Range-Doppler_fair comparison/3t4r_H104_blank_NewConfig/'
#h5_dir = '/home/pstudent/DeepSoli Models/gesture_data/Range-Angle_fair comparison/3t4r/scene_mixed/'
#h5_dir = '/home/pstudent/DeepSoli Models/gesture_data/transformto_h5/'

#x_test_blank = np.load(blank_dir + 'x_blank.npy')
#===========================================================
x_train_1 = np.load(h5_dir_1 + 'x_train.npy')
y_train_1 = np.load(h5_dir_1 + 'y_train.npy') 
#x_train_1_1 = np.load(h5_dir_1 + 'x_test.npy')
#y_train_1_1 = np.load(h5_dir_1 + 'y_test.npy') 

#x_train_1 = np.concatenate((x_train_1,x_train_1_1), axis=0)
#y_train_1 = np.concatenate((y_train_1,y_train_1_1), axis=0)

x_train_1 = np.transpose(x_train_1, (0,1,3,4,2))
#x_train_1 = x_train_1[:,:,:,32:,:]

x_test_1 = np.load(vali_dir_1 + 'x_test.npy')
y_test_1 = np.load(vali_dir_1 + 'y_test.npy') 
#x_train_1_1 = np.load(vali_dir_1 + 'x_test.npy')
#y_train_1_1 = np.load(vali_dir_1 + 'y_test.npy') 

#x_train_1 = np.concatenate((x_train_1,x_train_1_1), axis=0)
#y_train_1 = np.concatenate((y_train_1,y_train_1_1), axis=0)

x_test_1 = np.transpose(x_test_1, (0,1,3,4,2))
#x_test_1 = x_test_1[:,:,:,32:,:]

#===========================================================
x_train_1_original = np.load(h5_dir_1 + 'x_train.npy')
y_train_1_original = np.load(h5_dir_1 + 'y_train.npy') 

#x_test_blank = np.load(blank_dir + 'x_blank.npy')

x_test_1_original = np.load(h5_dir_1 + 'x_test.npy')
y_test_1_original = np.load(h5_dir_1 + 'y_test.npy') 
#print (x_test.shape)
#print (y_test.shape) 

x_train_1_original = np.transpose(x_train_1_original, (0,1,3,4,2))
x_test_1_original = np.transpose(x_test_1_original, (0,1,3,4,2))

#===========================================================

for i in range(0, 1440): #train as 80%
#    if i % 10 == 9 or i % 10 == 8 or i % 10 == 7 or i % 10 == 6 or i % 10 == 5:
#    if i % 10 == 9 or i % 10 == 8 or i % 10 == 7 or i % 10 == 6:
#    if i % 10 == 9 or i % 10 == 8 or i % 10 == 7:
    
    if i % 10 == 1 or i % 10 == 3 or i % 10 == 5 or i % 10 == 7 or i % 10 == 9:
#    if i % 10 == 9:
        x_train_1[i,...] = x_test_1_original[i,...]
        y_train_1[i,...] = y_test_1_original[i,...]
        x_test_1[i,...] = x_train_1_original[i,...]
        y_test_1[i,...] = y_train_1_original[i,...]           

        
del x_train_1_original, y_train_1_original, x_test_1_original, y_test_1_original

x_train_1 = x_train_1[:,:,:,:,:2]
#x_train = x_train[..., np.newaxis]
x_test_1 = x_test_1[:,:,:,:,:2]
#x_test = x_test[..., np.newaxis]


###########################################################################################################

x_train_2 = np.load(h5_dir_2 + 'x_train.npy')
y_train_2 = np.load(h5_dir_2 + 'y_train.npy') 
#x_train_2_1 = np.load(h5_dir_2 + 'x_test.npy')
#y_train_2_1 = np.load(h5_dir_2 + 'y_test.npy') 

#x_train_2 = np.concatenate((x_train_2,x_train_2_1), axis=0)
#y_train_2 = np.concatenate((y_train_2,y_train_2_1), axis=0)

x_train_2 = np.transpose(x_train_2, (0,1,3,4,2))
#x_train_2 = x_train_2[:,:,:,32:,:]


x_test_2 = np.load(vali_dir_2 + 'x_test.npy')
y_test_2 = np.load(vali_dir_2 + 'y_test.npy') 
#x_train_2_1 = np.load(vali_dir_2 + 'x_test.npy')
#y_train_2_1 = np.load(vali_dir_2 + 'y_test.npy') 

#x_train_2 = np.concatenate((x_train_2,x_train_2_1), axis=0)
#y_train_2 = np.concatenate((y_train_2,y_train_2_1), axis=0)

x_test_2 = np.transpose(x_test_2, (0,1,3,4,2))
#x_test_2 = x_test_2[:,:,:,32:,:]
#===========================================================
x_train_2_original = np.load(h5_dir_2 + 'x_train.npy')
y_train_2_original = np.load(h5_dir_2 + 'y_train.npy') 

#x_test_blank = np.load(blank_dir + 'x_blank.npy')

x_test_2_original = np.load(h5_dir_2 + 'x_test.npy')
y_test_2_original = np.load(h5_dir_2 + 'y_test.npy') 
#print (x_test.shape)
#print (y_test.shape) 

x_train_2_original = np.transpose(x_train_2_original, (0,1,3,4,2))
x_test_2_original = np.transpose(x_test_2_original, (0,1,3,4,2))

#===========================================================

for i in range(0, 1440): #train as 80%
#    if i % 10 == 9 or i % 10 == 8 or i % 10 == 7 or i % 10 == 6 or i % 10 == 5:
#    if i % 10 == 9 or i % 10 == 8 or i % 10 == 7 or i % 10 == 6:
#    if i % 10 == 9 or i % 10 == 8 or i % 10 == 7:
    
    if i % 10 == 1 or i % 10 == 3 or i % 10 == 5 or i % 10 == 7 or i % 10 == 9:
#    if i % 10 == 9:
        x_train_2[i,...] = x_test_2_original[i,...]
        y_train_2[i,...] = y_test_2_original[i,...]
        x_test_2[i,...] = x_train_2_original[i,...]
        y_test_2[i,...] = y_train_2_original[i,...]           

        
del x_train_2_original, y_train_2_original, x_test_2_original, y_test_2_original


#===========================================================

#x_train_1 = np.concatenate((x_train_1, x_test_1), axis=0)
#x_train_2 = np.concatenate((x_train_2, x_test_2), axis=0)
#y_train_1 = np.concatenate((y_train_1, y_test_1), axis=0)
#y_train_2 = np.concatenate((y_train_2, y_test_2), axis=0)
#
#vali_dir_1 = '/home/pstudent/DeepSoli Models/gesture_data/Range-Doppler_fair comparison/final data/3t4r_RDI/whole set as train/lab210_2/'
#vali_dir_2 = '/home/pstudent/DeepSoli Models/gesture_data/Range-Angle_fair comparison/final data/3t4r_RAI/whole set as train/lab210_2/'
#x_test_1 = np.load(vali_dir_1 + 'x_test.npy')
#y_test_1 = np.load(vali_dir_1 + 'y_test.npy') 
#x_test_2 = np.load(vali_dir_2 + 'x_test.npy')
#y_test_2 = np.load(vali_dir_2 + 'y_test.npy') 

#==============================================================================

#x_all = np.concatenate((x_train_1, x_test_1), axis=0)
#
#num_all = x_all.shape[0]
#width_all= x_all.shape[2]
#height_all = x_all.shape[3]
#channel_all = x_all.shape[4]
#
#x_all_mean_perChannel=np.zeros((num_all,64,width_all,height_all,channel_all))
#for i in range(0, channel_all):
#    temp = np.mean(x_all[:,:,:,:,i])
#    temp = np.repeat(temp, height_all, axis=0)
#    temp_2 = temp[np.newaxis,...]
#    temp_2 = np.repeat(temp_2, width_all, axis=0)
#    temp_2 = temp_2[np.newaxis,...]
#    temp_2 = np.repeat(temp_2, 64, axis=0)
#    temp_3 = temp_2[np.newaxis,...]
#    temp_3 = np.repeat(temp_3, num_all, axis=0)
#    x_all_mean_perChannel[:,:,:,:,i] = temp_3
#    print(i)
#
#x_all_std = np.zeros((1, channel_all))
#for j in range(0, channel_all):
#    x_all_std[:,j] = np.std(x_all[:,:,:,:,j])
#
#x_train_1 = (x_train_1 - x_all_mean_perChannel[:len(x_train_1),...])/x_all_std    
#x_test_1 = (x_test_1 - x_all_mean_perChannel[:len(x_test_1),...])/x_all_std
#
#del x_all_mean_perChannel, x_all_std , x_all
#del temp, temp_2, temp_3
#
#x_all = np.concatenate((x_train_2, x_test_2), axis=0)
#
#num_all = x_all.shape[0]
#width_all= x_all.shape[2]
#height_all = x_all.shape[3]
#channel_all = x_all.shape[4]
#
#x_all_mean_perChannel=np.zeros((num_all,64,width_all,height_all,channel_all))
#for i in range(0, channel_all):
#    temp = np.mean(x_all[:,:,:,:,i])
#    temp = np.repeat(temp, height_all, axis=0)
#    temp_2 = temp[np.newaxis,...]
#    temp_2 = np.repeat(temp_2, width_all, axis=0)
#    temp_2 = temp_2[np.newaxis,...]
#    temp_2 = np.repeat(temp_2, 64, axis=0)
#    temp_3 = temp_2[np.newaxis,...]
#    temp_3 = np.repeat(temp_3, num_all, axis=0)
#    x_all_mean_perChannel[:,:,:,:,i] = temp_3
#    print(i)
#
#x_all_std = np.zeros((1, channel_all))
#for j in range(0, channel_all):
#    x_all_std[:,j] = np.std(x_all[:,:,:,:,j])
#
#x_train_2 = (x_train_2 - x_all_mean_perChannel[:len(x_train_2),...])/x_all_std    
#x_test_2 = (x_test_2 - x_all_mean_perChannel[:len(x_test_2),...])/x_all_std
#
#del x_all_mean_perChannel, x_all_std , x_all
#del temp, temp_2, temp_3
########################################################################################
num_train = x_train_1.shape[0]
width_train = x_train_1.shape[2]
height_train = x_train_1.shape[3]
channel_train = x_train_1.shape[4]

x_train_mean_perChannel=np.zeros((num_train,64,width_train,height_train,channel_train))
for i in range(0, channel_train):
    temp = np.mean(x_train_1[:,:,:,:,i])
    temp = np.repeat(temp, height_train, axis=0)
    temp_2 = temp[np.newaxis,...]
    temp_2 = np.repeat(temp_2, width_train, axis=0)
    temp_2 = temp_2[np.newaxis,...]
    temp_2 = np.repeat(temp_2, 64, axis=0)
    temp_3 = temp_2[np.newaxis,...]
    temp_3 = np.repeat(temp_3, num_train, axis=0)
    x_train_mean_perChannel[:,:,:,:,i] = temp_3
    print(i)

x_train_std = np.zeros((1, channel_train))
for j in range(0, channel_train):
    x_train_std[:,j] = np.std(x_train_1[:,:,:,:,j])

x_train_1 = (x_train_1 - x_train_mean_perChannel)/x_train_std
x_test_1 = (x_test_1 - x_train_mean_perChannel)/x_train_std
del x_train_mean_perChannel, x_train_std
del temp, temp_2, temp_3
#==============================================================================
num_train = x_train_2.shape[0]
width_train = x_train_2.shape[2]
height_train = x_train_2.shape[3]
channel_train = x_train_2.shape[4]

x_train_mean_perChannel=np.zeros((num_train,64,width_train,height_train,channel_train))
for i in range(0, channel_train):
    temp = np.mean(x_train_2[:,:,:,:,i])
    temp = np.repeat(temp, height_train, axis=0)
    temp_2 = temp[np.newaxis,...]
    temp_2 = np.repeat(temp_2, width_train, axis=0)
    temp_2 = temp_2[np.newaxis,...]
    temp_2 = np.repeat(temp_2, 64, axis=0)
    temp_3 = temp_2[np.newaxis,...]
    temp_3 = np.repeat(temp_3, num_train, axis=0)
    x_train_mean_perChannel[:,:,:,:,i] = temp_3
    print(i)

x_train_std = np.zeros((1, channel_train))
for j in range(0, channel_train):
    x_train_std[:,j] = np.std(x_train_2[:,:,:,:,j])

x_train_2 = (x_train_2 - x_train_mean_perChannel)/x_train_std
x_test_2 = (x_test_2 - x_train_mean_perChannel)/x_train_std
del x_train_mean_perChannel, x_train_std
del temp, temp_2, temp_3

#==============================================================================
#model = load_model(model_dir + 'model_1t2r_lstm_parallel_batch12_ProgressSgd.h5')
#model.load_weights(model_dir + 'weights_1t2r_lstm_parallel_batch12_ProgressSgd.h5', by_name = True)
#score = model.evaluate(x_test, y_test, batch_size=12)
#predictions = model.predict(x_test) 
#==============================================================================
#y_train = np.utils.to_categorical(y_train, 7)
#y_test = np.utils.to_categorical(y_test, 7)

#datagen = ImageDataGenerator(
#        featurewise_center=True,
#        featurewise_std_normalization=True,
#        zoom_range=0.2)

#datagen.fit(x_train)

#x_train = x_train[..., np.newaxis]
#x_test = x_test[..., np.newaxis]
#gpu_options = tf.GPUOptions(allow_growth=True)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#
#tf.keras.backend.set_session(sess)


# =============================================================================
epochs=40
lr_power=0.9
lr_base=1e-2
def lr_scheduler(epoch):
#def lr_scheduler(epoch, mode='progressive_drops'):
     '''if lr_dict.has_key(epoch):
         lr = lr_dict[epoch]
         print 'lr: %f' % lr'''
 
#     if mode is 'power_decay':
#         # original lr scheduler
#         lr = lr_base * ((1 - float(epoch) / epochs) ** lr_power)
#     if mode is 'exp_decay':
#         # exponential decay
#         lr = (float(lr_base) ** float(lr_power)) ** float(epoch + 1)
#     # adam default lr
#     if mode is 'adam':
#         lr = 0.001
# 
#     if mode is 'progressive_drops':keras.layers
#         # drops as progression proceeds, good for sgd
#         if epoch > 0.9 * epochs:
#             lr = 1e-5
#         elif epoch > 0.75 * epochs:
#             lr = 1e-4
#         elif epoch > 0.5 * epochs:
#             lr = 1e-3
#         else:
#             lr = 1e-2
     if epoch > 0.9 * epochs:
        lr = 1e-5
     elif epoch > 0.75 * epochs:
        lr = 1e-4
     elif epoch > 0.5 * epochs:
        lr = 1e-3
     else:
        lr = 1e-2
     print('lr: %f' % lr)
     return lr
#     if epoch > 0.75 * epochs:
#        lr = 1e-5
#     elif epoch > 0.5 * epochs:
#        lr = 1e-4
#     elif epoch > 0.25 * epochs:
#        lr = 1e-3
#     else:
#        lr = 1e-2
#     print('lr: %f' % lr)
#     return lr
 
scheduler = LearningRateScheduler(lr_scheduler)
# =============================================================================

# =============================================================================
#x_train = x_train.astype('float32') / 255. - 0.5
#x_test = x_test.astype('float32') / 255. - 0.5
# =============================================================================

os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
with tf.device('/gpu:2'):
    input_1 = Input(shape=x_train_1.shape[1:])
    x = TimeDistributed(Conv2D(32, (3, 3), strides=1, padding='valid'), input_shape=x_train_1.shape[1:])(input_1)
    #TimeDistributed(MaxPooling2D((3, 3), strides=1, padding='valid', data_format='channels_last')))
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Activation('relu'))(x)
#    TimeDistributed(Dropout(0.4)))
    
    x = TimeDistributed(Conv2D(64, (3, 3)))(x)
    #TimeDistributed(MaxPooling2D((3, 3), strides=1, padding='valid', data_format='channels_last')))
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Activation('relu'))(x)
    x = TimeDistributed(Dropout(0.4))(x)
    
    x = TimeDistributed(Conv2D(128, (3, 3)))(x)
    #TimeDistributed(MaxPooling2D((3, 3), strides=1, padding='valid', data_format='channels_last')))
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Activation('relu'))(x)
    x = TimeDistributed(Dropout(0.4))(x)
    
    x = TimeDistributed(Flatten())(x)
    

#    x = Model(inputs=input_1, outputs=x)    
    
with tf.device('/gpu:3'):
    input_2 = Input(shape=x_train_2.shape[1:])
    y = TimeDistributed(Conv2D(32, (3, 3), strides=1, padding='valid'), input_shape=x_train_2.shape[1:])(input_2)
    #TimeDistributed(MaxPooling2D((3, 3), strides=1, padding='valid', data_format='channels_last')))
    y = TimeDistributed(BatchNormalization())(y)
    y = TimeDistributed(Activation('relu'))(y)
#    TimeDistributed(Dropout(0.4)))
    
    y = TimeDistributed(Conv2D(64, (3, 3)))(y)
    #TimeDistributed(MaxPooling2D((3, 3), strides=1, padding='valid', data_format='channels_last')))
    y = TimeDistributed(BatchNormalization())(y)
    y = TimeDistributed(Activation('relu'))(y)
    y = TimeDistributed(Dropout(0.4))(y)
    
    y = TimeDistributed(Conv2D(128, (3, 3)))(y)
    #TimeDistributed(MaxPooling2D((3, 3), strides=1, padding='valid', data_format='channels_last')))
    y = TimeDistributed(BatchNormalization())(y)
    y = TimeDistributed(Activation('relu'))(y)
    y = TimeDistributed(Dropout(0.4))(y)
    
    y = TimeDistributed(Flatten())(y)
    

#    y = Model(inputs=input_2, outputs=y)       
    
with tf.device('/cpu:0'):
    combined = concatenate([x, y])
    z = TimeDistributed(Dense(512))(combined)
    #    x = TimeDistributed(BatchNormalization())(x)
    z = TimeDistributed(Activation('relu'))(z)
    z = TimeDistributed(Dropout(0.5))(z)
    
    z = TimeDistributed(Dense(512))(z)
    #    x = TimeDistributed(BatchNormalization())(x)
    z = TimeDistributed(Activation('relu'))(z)
    z = TimeDistributed(Dropout(0.5))(z)
    #    model.add(TimeDistributed(Dropout(0.5)))
    #model.add(TimeDistributed(Lambda(lambda x: tf.expand_dims(model.output, axis=-1))))
    #    model.add(Bidirectional(LSTM(512, return_sequences=True)))
    #    Att_Output = Bidirectional(GRU(512, return_sequences=True))(x)
    
#    z = LSTM(512, activation='sigmoid', return_sequences=True)(z)   
#    z = TimeDistributed(Dropout(0.5))(z)
    
#    z = TimeDistributed(Dense(512))(combined)
#    #    x = TimeDistributed(BatchNormalization())(x)
#    z = TimeDistributed(Activation('relu'))(z)
#    z = TimeDistributed(Dropout(0.5))(z)
#    
#    z = TimeDistributed(Dense(512))(z)
#    #    x = TimeDistributed(BatchNormalization())(x)
#    z = TimeDistributed(Activation('relu'))(z)
#    z = TimeDistributed(Dropout(0.5))(z)
#    #    model.add(TimeDistributed(Dropout(0.5)))
#    #model.add(TimeDistributed(Lambda(lambda x: tf.expand_dims(model.output, axis=-1))))
#    #    model.add(Bidirectional(LSTM(512, return_sequences=True)))
#    #    Att_Output = Bidirectional(GRU(512, return_sequences=True))(x)
#    z = Bidirectional(LSTM(512, return_sequences=True))(z)   
    z = Bidirectional(LSTM(512, return_sequences=True))(z)   
    z = TimeDistributed(Dropout(0.5))(z)
#    
    z = TimeDistributed(Dense(12))(z)
    pred = TimeDistributed(Activation('softmax'))(z)


########################################################################################################
#model.save(h5_dir + 'model_uni-lstm_RangeAngle_64x256_255normalized_12label_80-20_batch12_ProgressSgd.h5')
#model.save_weights(h5_dir + 'weights_uni-lstm_RangeAngle_64x256_255normalized_12label_80-20_batch12_ProgressSgd.h5')
model = Model(inputs=[input_1, input_2], outputs=[pred])
parallel_model = multi_gpu_model(model, gpus=2)

#nadam = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

################################################################model.save(h5_dir + 'model_uni-lstm_RangeAngle_64x256_255normalized_12label_80-20_batch12_ProgressSgd.h5')
#model.save_weights(h5_dir + 'weights_uni-lstm_RangeAngle_64x256_255normalized_12label_80-20_batch12_ProgressSgd.h5')

sgd = optimizers.SGD(lr=0.0, momentum=0.9, decay=0.0, nesterov=False)
parallel_model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
parallel_model.summary()

history = parallel_model.fit([x_train_1, x_train_2], [y_train_1], 
          batch_size=12, epochs=40, shuffle=True, verbose=1, 
          callbacks=[scheduler],
          validation_data = ([x_test_1, x_test_2], [y_test_1])
          )

#score = model.evaluate(x_test, y_test, batch_size=12)

#history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=7),
#                              epochs=60, shuffle=True, verbose=1, 
##          callbacks=[scheduler],
#          validation_data = (x_test, y_test))

#model.fit(x_train, y_train, batch_size=7, epochs=40, shuffle=True, verbose=1)

predictions = parallel_model.predict([x_test_1, x_test_2])
#predictions_classes = model.predict_classes(x_test)

##################################################################################
#


#
##print(history.history.keys())
#
#################################################################################

######################################################
#


######################################################

#parallel_model = multi_gpu_model(model, gpus=2)
#    
#parallel_model.compile(optimizer=sgd,
#          loss='categorical_crossentropy',
#          metrics=['accuracy'])
#parallel_model.summary()
#
#history = parallel_model.fit(x_train, y_train, 
#          batch_size=28, epochs=60, shuffle=True, verbose=1, 
#          callbacks=[scheduler],
#          validation_data = (x_test, y_test))
#
#score = parallel_model.evaluate(x_test, y_test, batch_size=28)
#
##parallel_model.save('model_0530_all reversed2 in trainset.h5')
##parallel_model.save_weights('weights_0530_all reversed2 in trainset.h5')
#
#predictions = parallel_model.predict(x_test)

######################################################
#y_pred = (predictions > 0.5) 
#y_p=(y_pred>0).astype(int)
#######################################################
#xx=[]
#yy=[]
#yy_temp=[]

######################################################

#label=[1,2,3,4,5,6,7]
#
#for i in range(0, len(y_test)):
#    for j in range(0, 7):
#       if y_test[i,1,j] ==1 :
#           xx.append(j+1)
#           
#for i in range(0, len(y_test)):
#    y_p_slice = y_p[i,:,:]
#    yy_temp=[]
#    for j in range(0, 64):
#        for k in range(0, 7):            
#            if y_p_slice[j,k] == 1 :    
#               yy_temp.append(k+1)
#               label_of_frame = max(set(yy_temp), key=yy_temp.count)               
#
#    yy.append(label_of_frame)    
    
######################################################  

#label=[1,2,3,4,5,6,7,8,9,10,11,12]
#
#for i in range(0, len(y_test)):
#    for j in range(0, 12):
#       if y_test[i,1,j] ==1 :
#           xx.append(j+1)
#           
#for i in range(0, len(y_test)):
#    y_p_slice = y_p[i,:,:]
#    yy_temp=[]
#    for j in range(0, 64):
#        for k in range(0, 12):            
#            if y_p_slice[j,k] == 1 :    
#               yy_temp.append(k+1)
#               label_of_frame = max(set(yy_temp), key=yy_temp.count)               
#
#    yy.append(label_of_frame)    
#    
#######################################################
###matrix = metrics.confusion_matrix(y_test.argmax(axis=1), y_p.argmax(axis=1))
#mat = metrics.confusion_matrix(xx, yy,labels=label)
### =============================================================================
#plt.matshow(mat)
#plt.colorbar()
#plt.xlabel('predicted')
#plt.ylabel('answer')
#plt.xticks(np.arange(mat.shape[1]),label)
#plt.yticks(np.arange(mat.shape[1]),label)
#plt.show()
# =============================================================================
#mat_percentage =mat.astype(float)
########################################################

#for i in range(0,7):
#       for j in range(0,7):
#            s = float(sum(mat[i,:]))
#            mat_percentage[i,j]= mat_percentage[i,j]/s

########################################################

#for i in range(0, 12):
#       for j in range(0,12):
#            s = float(sum(mat[i,:]))
#            mat_percentage[i,j]= mat_percentage[i,j]/s
            
########################################################
#mat_percentage = []
#for i in range(0, len(predictions[:,1,1])):
#    for j in range(0, 12):
#        mat_percentage[] = sum(predictions[i,:,j])

########################################################

#mat_percentage =mat.astype(float)
########################################################

#for i in range(0,7):#frames_prob_sum=[]
#temp=[]
#mat_percentage = np.zeros((12, 12))
#
#for i in range(0, 64):
#    frames_prob_sum = np.sum(predictions, axis=1) # sum as (test_num, label)
#
#count=0;
#
################## ladder-shaped 1-to-7 label###########
##for j in range(0, 24):
##    if j==0:
##        temp = frames_prob_sum[count:count+7,:]
##    else:
##        temp = temp + frames_prob_sum[count:count+7,:]
##    count+=7
##mat_percentage[0:7, :] = temp/24
########################################################
#for j in range(0, 7):      
#    temp = sum(frames_prob_sum[count:count+24,:])
#    mat_percentage[j, :] = temp/24
#    count+=24
#
#for j in range(7, 10):
#    temp = sum(frames_prob_sum[count:count+12,:])
#    mat_percentage[j, :] = temp/12
#    count=count+12
#
#for j in range(10, 12):
#    temp = sum(frames_prob_sum[count:count+32,:])
#    mat_percentage[j, :] = temp/32
#    count=count+32
#    
#mat_percentage = mat_percentage/64
#
#print('The vali accuracy derived from confusion matrix is:\n')
#print(np.trace(mat_percentage)/12)
#       for j in range(0,7):
#            s = float(sum(mat[i,:]))
#            mat_percentage[i,j]= mat_percentage[i,j]/s

########################################################

#for i in range(0, 12):
#       for j in range(0,12):
#            s = float(sum(mat[i,:]))
#            mat_percentage[i,j]= mat_percentage[i,j]/s
            
########################################################
#==============================================================================
# mat_percentage = np.zeros((12, 12))
# for i in range(0, 24*7):
#     for j in range(0, 7):        
#         mat_percentage[i%7, j] = sum(predictions[i,:,j])/64
# 
# 
# for i in range(24*7, 24*7 + 10):
#     for j in range(0, 12):
#         mat_percentage[7, j] = sum(predictions[i,:,j])/64
#     
# for i in range(24*7 + 10, 24*7 + 20):
#     for j in range(0, 12):
#         mat_percentage[8, j] = sum(predictions[i,:,j])/64
#     
# for i in range(24*7 + 20, 24*7 + 30):
#     for j in range(0, 12):
#         mat_percentage[9, j] = sum(predictions[i,:,j])/64
#         
# for i in range(24*7 + 30, 24*7 + 62):
#     for j in range(0, 12):
#         mat_percentage[10, j] = sum(predictions[i,:,j])/64
# 
# for i in range(24*7 + 62, 24*7 + 94):
#     for j in range(0, 12):
#         mat_percentage[11, j] = sum(predictions[i,:,j])/64
#==============================================================================
## confusion for scene-mixing
            
#frames_prob_sum=[]
#temp=[]
#mat_percentage = np.zeros((12, 12))
#
#for i in range(0, 64):
#    frames_prob_sum = np.sum(predictions, axis=1) # sum as (test_num, label)
#
#count=0;
#for j in range(0, 7):      
#    temp = sum(frames_prob_sum[count:count+42,:])
#    mat_percentage[j, :] = temp/42
#    count+=42       
#
#for j in range(7, 10):
#    temp = sum(frames_prob_sum[count:count+24,:])
#    mat_percentage[j, :] = temp/24
#    count+=24
#
#for j in range(10, 12):
#    temp = sum(frames_prob_sum[count:count+52,:])
#    mat_percentage[j, :] = temp/52
#    count+=52
#    
#mat_percentage = mat_percentage/64
#print('\nThe vali accuracy derived from confusion matrix is:\n')
#print(np.trace(mat_percentage)/12)



##############################################################
## confusion for lab210

#frames_prob_sum=[]
#temp=[]
#mat_percentage = np.zeros((12, 12))
#
#for i in range(0, 64):
#    frames_prob_sum = np.sum(predictions, axis=1) # sum as (test_num, label)
#
#count=0;
#
################## ladder-shaped 1-to-7 label###########
##for j in range(0, 24):
##    if j==0:
##        temp = frames_prob_sum[count:count+7,:]
##    else:
##        temp = temp + frames_prob_sum[count:count+7,:]
##    count+=7
##mat_percentage[0:7, :] = temp/24
########################################################
#for j in range(0, 7):      
#    temp = sum(frames_prob_sum[count:count+24,:])
#    mat_percentage[j, :] = temp/24
#    count+=24
#
#for j in range(7, 10):
#    temp = sum(frames_prob_sum[count:count+12,:])
#    mat_percentage[j, :] = temp/12
#    count=count+12
#
#for j in range(10, 12):
#    temp = sum(frames_prob_sum[count:count+32,:])
#    mat_percentage[j, :] = temp/32
#    count=count+32
#    
#mat_percentage = mat_percentage/64
#
#print('The vali accuracy derived from confusion matrix is:\n')
#print(np.trace(mat_percentage)/12)
#np.save(h5_dir + 'mat_percentage', mat_percentage)
#########################################################

##confusion for H104

#frames_prob_sum=[]
#temp=[]
#mat_percentage = np.zeros((12, 12))
#
#for i in range(0, 64):
#    frames_prob_sum = np.sum(predictions, axis=1) # sum as (test_num, label)
#
#count=0;
#for j in range(0, 7):      
#    temp = sum(frames_prob_sum[count:count+18,:])
#    mat_percentage[j, :] = temp/18
#    count+=18        
#
#for j in range(7, 10):
#    temp = sum(frames_prob_sum[count:count+12,:])
#    mat_percentage[j, :] = temp/12
#    count+=12
#
#for j in range(10, 12):
#    temp = sum(frames_prob_sum[count:count+20,:])
#    mat_percentage[j, :] = temp/20
#    count+=20
#    
#mat_percentage = mat_percentage/64
#print('The vali accuracy derived from confusion matrix is:\n')
#print(np.trace(mat_percentage)/12)

#==============================================================================
# confusion for lab210 other points
#frames_prob_sum=[]
#temp=[]
#mat_percentage = np.zeros((12, 12))
#
#for i in range(0, 64):
#    frames_prob_sum = np.sum(predictions, axis=1) # sum as (test_num, label)
#
#count=0;
#for j in range(0, 12):      
#    temp = sum(frames_prob_sum[count:count+20,:])
#    mat_percentage[j, :] = temp/20
#    count+=20        
#    
#mat_percentage = mat_percentage/64
#print('The vali accuracy derived from confusion matrix is:\n')
#print(np.trace(mat_percentage)/12)



#==============================================================================
## confusion matrix for H210 whole testing

#frames_prob_sum=[]
#temp=[]
#mat_percentage = np.zeros((12, 12))
#
#for i in range(0, 64):
#    frames_prob_sum = np.sum(predictions, axis=1) # sum as (test_num, label)
#
#count=0;
#for j in range(0, 7):      
#    temp = sum(frames_prob_sum[count:count+120,:])
#    mat_percentage[j, :] = temp/120
#    count+=120       
#
#for j in range(7, 10):
#    temp = sum(frames_prob_sum[count:count+60,:])
#    mat_percentage[j, :] = temp/60
#    count+=60
#
#for j in range(10, 12):
#    temp = sum(frames_prob_sum[count:count+160,:])
#    mat_percentage[j, :] = temp/160
#    count+=160
#    
#mat_percentage = mat_percentage/64
#print('The vali accuracy derived from confusion matrix is:\n')
#print(np.trace(mat_percentage)/12)

#==============================================================================
##confusion matrix for H104 whole testing

#frames_prob_sum=[]
#temp=[]
#mat_percentage = np.zeros((12, 12))
#
#for i in range(0, 64):
#    frames_prob_sum = np.sum(predictions, axis=1) # sum as (test_num, label)
#
#count=0;
#for j in range(0, 7):      
#    temp = sum(frames_prob_sum[count:count+90,:])
#    mat_percentage[j, :] = temp/90
#    count+=90       
#
#for j in range(7, 10):
#    temp = sum(frames_prob_sum[count:count+60,:])
#    mat_percentage[j, :] = temp/60
#    count+=60
#
#for j in range(10, 12):
#    temp = sum(frames_prob_sum[count:count+100,:])
#    mat_percentage[j, :] = temp/100
#    count+=100
#    
#mat_percentage = mat_percentage/64
#print('The vali accuracy derived from confusion matrix is:\n')
#print(np.trace(mat_percentage)/12)
#==============================================================================


##confusion matrix for final data

#frames_prob_sum=[]
#temp=[]
#mat_percentage = np.zeros((12, 12))
#
#for i in range(0, 64):
#    frames_prob_sum = np.sum(predictions, axis=1) # sum as (test_num, label)
#
#count=0;
#for j in range(0, 7):      
#    temp = sum(frames_prob_sum[count:count+42,:])
#    mat_percentage[j, :] = temp/42
#    count+=42       
#
#for j in range(7, 10):
#    temp = sum(frames_prob_sum[count:count+24,:])
#    mat_percentage[j, :] = temp/24
#    count+=24
#
#for j in range(10, 12):
#    temp = sum(frames_prob_sum[count:count+24,:])
#    mat_percentage[j, :] = temp/24
#    count+=24
#    
#mat_percentage = mat_percentage/64
#print('The vali accuracy derived from confusion matrix is:\n')
#print(np.trace(mat_percentage)/12)

#==============================================================================

frames_prob_sum=[]

mat_percentage = np.zeros((12, 12))

for i in range(0, 64):
    frames_prob_sum = np.sum(predictions, axis=1) # sum as (test_num, label)

count=0;
for j in range(0, 12):
    
    temp_labelSum=[]
    temp=np.zeros((1, 12))
    
    temp_labelSum = sum(frames_prob_sum[count:count+10,:])
    count+=10
    temp = temp_labelSum
    mat_percentage[j, :] = temp
    
mat_percentage = mat_percentage/(64*10)
print('The vali accuracy derived from confusion matrix is:\n')
print(np.trace(mat_percentage)/12)

########################################################
#tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
#X_tsne = tsne.fit_transform(xx)
#
#x_min, x_max = X_tsne.min(0), X_tsne.max(0)
#X_norm = (X_tsne - x_min) / (x_max - x_min)  
#plt.figure(figsize=(8, 8))
#for i in range(X_norm.shape[0]):
#    plt.text(X_norm[i, 0], X_norm[i, 1], str(yy[i]), color=plt.cm.Set1(yy[i]), 
#             fontdict={'weight': 'bold', 'size': 9})
#plt.xticks([])
#plt.yticks([])
#plt.show()
output_dir = '/home/lab210/MattYen_workspace/gest_August/original_cases/phase_3/early/1t2r/bilstm/'
#
# summarize history for accuracy
np.save(output_dir+'mat_percentage.npy', mat_percentage)
np.save(output_dir+'predictions.npy', predictions)
# summarize history for accuracy
fig_acc = plt.gcf()
plt.plot(history.history['acc'], linestyle='--')
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.draw()

fig_acc.savefig(output_dir +'acc.png')
#fig_acc.savefig(output_dir +'acc.png')

# summarize history for loss
fig_loss = plt.gcf()
plt.plot(history.history['loss'], linestyle='--')
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.draw()

fig_loss.savefig(output_dir + 'loss.png')

parallel_model.save(output_dir  + 'model_1t2r_bilstm_early_gaussian_batch12_ProgressSgd.h5')
parallel_model.save_weights(output_dir  + 'weights_1t2r_bilstm_early_gaussian_batch12_ProgressSgd.h5')


#==============================================================================
#model = load_model(output_dir  + 'model_1t2r_lstm_early_gaussian_batch12_ProgressSgd.h5')
#model.load_weights(output_dir  + 'weights_1t2r_lstm_early_gaussian_batch12_ProgressSgd.h5', by_name = True)
#score = model.evaluate([x_test_1, x_test_2], [y_test_1, y_test_2], batch_size=12)
#predictions = model.predict([x_test_1, x_test_2]) 
#
#mat_percentage = np.load(output_dir+'mat_percentage.npy')
#print('The vali accuracy derived from confusion matrix is:\n')
#print(np.trace(mat_percentage)/12)
#==============================================================================