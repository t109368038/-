from __future__ import print_function
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dropout, Flatten, Dense, LSTM, TimeDistributed, Bidirectional
from tensorflow.keras.layers import MaxPooling2D
#from tensorflow.keras.layers import GRU
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#import tensorflow.distribute.MirroredStrategy as strategy
import tensorflow.keras.optimizers as optimizers
from sklearn import metrics
import numpy as np
from collections import Counter
from sklearn.metrics import confusion_matrix
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import LearningRateScheduler
#from gesture_LearningRateScheduler import scheduler

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

#h5_dir='/home/pstudent/DeepSoli Models/gesture_data/64x32_normalized/npy_random_80-20/'
h5_dir='/home/pstudent/DeepSoli Models/gesture_data/RangeAngle/64x64_255normalized_12label/60-40/'

x_train = np.load(h5_dir + 'x_train.npy')
y_train = np.load(h5_dir + 'y_train.npy') 
print (x_train.shape) 
print (y_train.shape) 

x_test = np.load(h5_dir + 'x_test.npy')
y_test = np.load(h5_dir + 'y_test.npy') 
print (x_test.shape)
print (y_test.shape) 

x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

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
epochs=60
#lr_power=0.9
#lr_base=1e-2
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
#     if mode is 'progressive_drops':
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
 
scheduler = LearningRateScheduler(lr_scheduler)
# =============================================================================


# =============================================================================
#x_train = x_train.astype('float32') / 255. - 0.5
#x_test = x_test.astype('float32') / 255. - 0.5
# =============================================================================
with tf.device('/gpu:0'):
    model = Sequential()
    
    model.add(TimeDistributed(Conv2D(32, (9, 9), strides=1, padding='valid'), input_shape=x_train.shape[1:]))
#    model.add(TimeDistributed(MaxPooling2D((3, 3), strides=1, padding='valid', data_format='channels_last')))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Conv2D(64, (9, 9), activation='relu')))
#    model.add(TimeDistributed(MaxPooling2D((3, 3), strides=1, padding='valid', data_format='channels_last')))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Dropout(0.4)))
    model.add(TimeDistributed(Conv2D(128, (9, 9), activation='relu')))
#    model.add(TimeDistributed(MaxPooling2D((3, 3), strides=1, padding='valid', data_format='channels_last')))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Dropout(0.4)))  
    
    
    model.add(TimeDistributed(Flatten()))
    model.add(TimeDistributed(Dense(512, activation='relu')))
    #model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Dropout(0.5)))
    model.add(TimeDistributed(Dense(512, activation='relu')))
    #model.add(TimeDistributed(BatchNormalization()))
    
    #model.add(TimeDistributed(Lambda(lambda x: tf.expand_dims(model.output, axis=-1))))
    model.add(Bidirectional(LSTM(512, return_sequences=True)))
    #model.add(TimeDistributed(BatchNormalization()))
    
    model.add(TimeDistributed(Dropout(0.5)))
    model.add(TimeDistributed(Dense(12, activation='softmax')))

sgd = optimizers.SGD(lr=0.0, momentum=0.9, decay=0.0, nesterov=False)
#nadam = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)


################################################################

model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

history = model.fit(x_train, y_train, 
          batch_size=7, epochs=60, shuffle=True, verbose=1, 
          callbacks=[scheduler],
          validation_data = (x_test, y_test))

score = model.evaluate(x_test, y_test, batch_size=7)

#history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=7),
#                              epochs=60, shuffle=True, verbose=1, 
##          callbacks=[scheduler],
#          validation_data = (x_test, y_test))

#model.fit(x_train, y_train, batch_size=14, epochs=40, shuffle=True, verbose=1)

predictions = model.predict(x_test)
##################################################################################
#
##model = load_model('/home/pstudent/DeepSoli Models/gesture_data/NormalAndReverseMixed/80-20/sgd/batch 11/model_NormalAndReverseMixed_80-20_batch11_sgd_type4.h5')

#
##print(history.history.keys())
#
#################################################################################

#model.save(h5_dir + 'model_RangeAngle_64x64_normalized_10label_60-40_batch7_sgd.h5')
#model.save_weights(h5_dir + 'weights_RangeAngle_64x64_normalized_10label_60-40_batch7_sgd.h5')

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
#          batch_size=14, epochs=60, shuffle=True, verbose=1, 
##          callbacks=[scheduler],
#          validation_data = (x_test, y_test))
#
#score = parallel_model.evaluate(x_test, y_test, batch_size=14)
#
##parallel_model.save('model_0530_all reversed2 in trainset.h5')
##parallel_model.save_weights('weights_0530_all reversed2 in trainset.h5')
#
#predictions = parallel_model.predict(x_test)

######################################################

y_pred = (predictions > 0.5) 
y_p=(y_pred>0).astype(int)

xx=[]
yy=[]
yy_temp=[]

######################################################

#label=[1,2,3,4,5,6,7]
#
#for i in range(0, len(y_test)):
#    for j in range(0,7):
#       if y_test[i,1,j] ==1 :
#           xx.append(j+1)
#           
#for i in range(0, len(y_test)):
#    y_p_slice = y_p[i,:,:]
#    for j in range(0, 64):
#        for k in range(0, 7):
#            yy_temp=[]
#            if y_p_slice[j,k] == 1 :    
#               yy_temp.append(k+1)
#               label_of_frame = max(set(yy_temp), key=yy_temp.count)               
#    yy.append(label_of_frame)  
    
######################################################  

#label=[1,2,3,4,5,6,7,8,9,10]
#
#for i in range(0, len(y_test)):
#    for j in range(0, 10):
#       if y_test[i,1,j] ==1 :
#           xx.append(j+1)
#           
#for i in range(0, len(y_test)):
#    y_p_slice = y_p[i,:,:]
#    yy_temp=[]
#    for j in range(0, 64):
#        for k in range(0, 10):            
#            if y_p_slice[j,k] == 1 :    
#               yy_temp.append(k+1)
#               label_of_frame = max(set(yy_temp), key=yy_temp.count)               
#
#    yy.append(label_of_frame)    
    
######################################################

label=[1,2,3,4,5,6,7,8,9,10,11,12]

for i in range(0, len(y_test)):
    for j in range(0, 12):
       if y_test[i,1,j] ==1 :
           xx.append(j+1)
           
for i in range(0, len(y_test)):
    y_p_slice = y_p[i,:,:]
    yy_temp=[]
    for j in range(0, 64):
        for k in range(0, 12):            
            if y_p_slice[j,k] == 1 :    
               yy_temp.append(k+1)
               label_of_frame = max(set(yy_temp), key=yy_temp.count)               

    yy.append(label_of_frame)       
#########################################################
##matrix = metrics.confusion_matrix(y_test.argmax(axis=1), y_p.argmax(axis=1))
mat = metrics.confusion_matrix(xx, yy,labels=label)
## =============================================================================
plt.matshow(mat)
plt.colorbar()
plt.xlabel('predicted')
plt.ylabel('answer')
plt.xticks(np.arange(mat.shape[1]),label)
plt.yticks(np.arange(mat.shape[1]),label)
plt.show()
# =============================================================================
mat_percentage =mat.astype(float)
########################################################

#for i in range(0,7):
#       for j in range(0,7):
#            s = float(sum(mat[i,:]))
#            mat_percentage[i,j]= mat_percentage[i,j]/s

########################################################

#for i in range(0, 10):
#       for j in range(0,10):
#            s = float(sum(mat[i,:]))
#            mat_percentage[i,j]= mat_percentage[i,j]/s
            
########################################################

#for i in range(0, 12):
#       for j in range(0,12):
#            s = float(sum(mat[i,:]))
#            mat_percentage[i,j]= mat_percentage[i,j]/s
            
########################################################
            
            
            


#
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#plt.savefig(h5_dir +'acc.png')

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#plt.savefig(h5_dir + 'loss.png')

