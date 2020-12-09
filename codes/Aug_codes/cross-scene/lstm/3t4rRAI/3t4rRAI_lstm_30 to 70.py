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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


h5_dir='/home/lab210/MattYen_workspace/gest_August/cross-scene/3t4rRAI/'


x_train = np.load(h5_dir + 'x_train.npy')
y_train = np.load(h5_dir + 'y_train.npy') 

#x_test_blank = np.load(blank_dir + 'x_blank.npy')

x_test = np.load(h5_dir + 'x_test.npy')
y_test = np.load(h5_dir + 'y_test.npy') 
#print (x_test.shape)
#print (y_test.shape) 

x_train = np.transpose(x_train, (0,1,3,4,2))
x_test = np.transpose(x_test, (0,1,3,4,2))

#============================================================
x_train_new = np.load(h5_dir + 'x_train.npy')
y_train_new = np.load(h5_dir + 'y_train.npy') 


x_test_new = np.load(h5_dir + 'x_test.npy')
y_test_new = np.load(h5_dir + 'y_test.npy') 

x_train_new = np.transpose(x_train_new, (0,1,3,4,2))
x_test_new = np.transpose(x_test_new, (0,1,3,4,2))

for i in range(0, 1440): #train as 80%
#    if i % 10 == 9 or i % 10 == 8 or i % 10 == 7 or i % 10 == 6 or i % 10 == 5:
#    if i % 10 == 9 or i % 10 == 8 or i % 10 == 7 or i % 10 == 6:
#    if i % 10 == 9 or i % 10 == 8 or i % 10 == 7:
    
    if i % 10 == 0 or i % 10 == 1 or i % 10 == 2 or i % 10 == 3 or i % 10 == 4 or i % 10 == 9 or i % 10 == 8: 
#    if i % 10 == 9:
        x_train_new[i,...] = x_test[i,...]
        y_train_new[i,...] = y_test[i,...]
#        x_train_new = np.delete(x_train_new, i, 0)
#        y_train_new = np.delete(y_train_new, i, 0)
        
for i in range(0, 1440): #test as 20%
#    if i % 10 == 0 or i % 10 == 1 or i % 10 == 2 or i % 10 == 3 or i % 10 == 4 or i % 10 == 5 or i % 10 == 6 or i % 10 == 7: 
    if i % 10 == 5 or i % 10 == 6 or i % 10 == 7  :
        x_test_new[i,...] = x_train[i,...]
        y_test_new[i,...] = y_train[i,...]
#        x_test_new = np.delete(x_test_new, i, 0)
#        y_test_new = np.delete(y_test_new, i, 0)
        
del x_train, y_train, x_test, y_test
#del temp_train_x, temp_train_y, temp_test_x, temp_test_y

#

num_train = x_train_new.shape[0]
width_train = x_train_new.shape[2]
height_train = x_train_new.shape[3]
channel_train = x_train_new.shape[4]

x_train_mean_perChannel=np.zeros((num_train,64,width_train,height_train,channel_train))
for i in range(0, channel_train):
    temp = np.mean(x_train_new[:,:,:,:,i])
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
    x_train_std[:,j] = np.std(x_train_new[:,:,:,:,j])

x_train_new = (x_train_new - x_train_mean_perChannel)/x_train_std

################################################################
x_test_new = (x_test_new - x_train_mean_perChannel)/x_train_std
################################################################

del x_train_mean_perChannel, x_train_std
del temp, temp_2, temp_3
# =============================================================================
epochs=40
lr_power=0.9
lr_base=1e-2
def lr_scheduler(epoch):
#def lr_scheduler(epoch, mode='progressive_drops'):
     '''if lr_dict.has_key(epoch):
         lr = lr_dict[epoch]
         print 'lr: %f' % lr'''
 
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
    
    model.add(TimeDistributed(Conv2D(32, (3, 3), strides=1, padding='valid'), input_shape=x_train_new.shape[1:]))
    #model.add(TimeDistributed(MaxPooling2D((3, 3), strides=1, padding='valid', data_format='channels_last')))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
#    model.add(TimeDistributed(Dropout(0.4)))
    
    model.add(TimeDistributed(Conv2D(64, (3, 3))))
    #model.add(TimeDistributed(MaxPooling2D((3, 3), strides=1, padding='valid', data_format='channels_last')))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Dropout(0.4)))
    
    model.add(TimeDistributed(Conv2D(128, (3, 3))))
    #model.add(TimeDistributed(MaxPooling2D((3, 3), strides=1, padding='valid', data_format='channels_last')))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Dropout(0.4)))
    
    model.add(TimeDistributed(Flatten()))
    
    model.add(TimeDistributed(Dense(512)))
#    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Dropout(0.5)))
    
    model.add(TimeDistributed(Dense(512)))
#    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Dropout(0.5)))    
#    model.add(TimeDistributed(Dropout(0.5)))
    #model.add(TimeDistributed(Lambda(lambda x: tf.expand_dims(model.output, axis=-1))))
    model.add(Bidirectional(LSTM(512, return_sequences=True)))
#    model.add(LSTM(512, return_sequences=True))
    model.add(TimeDistributed(Dropout(0.5))) 
#    model.add(LSTM(512, return_sequences=True))
    #model.add(TimeDistributed(BatchNormalization()))
    
#    model.add(TimeDistributed(SeqSelfAttention(attention_activation='sigmoid')))
    model.add(TimeDistributed(Dense(12)))
#    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('softmax')))


#nadam = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)


################################################################model.save(h5_dir + 'model_uni-lstm_RangeAngle_64x256_255normalized_12label_80-20_batch12_ProgressSgd.h5')
#model.save_weights(h5_dir + 'weights_uni-lstm_RangeAngle_64x256_255normalized_12label_80-20_batch12_ProgressSgd.h5')

sgd = optimizers.SGD(lr=0.0, momentum=0.9, decay=0.0, nesterov=False)
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

history = model.fit(x_train_new, y_train_new, 
          batch_size=12, epochs=40, shuffle=True, verbose=1, 
          callbacks=[scheduler],
          validation_data = (x_test_new, y_test_new)
          )

predictions = model.predict(x_test_new)

#==============================================================================
# Aug. NewConfig data_cross-scene

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
    temp = temp + temp_labelSum
    mat_percentage[j, :] = temp
    
mat_percentage = mat_percentage/(64*10)
print('The vali accuracy derived from confusion matrix is:\n')
print(np.trace(mat_percentage)/12)

#------------------------------------------------------------
#add per frame probablity
frame_prob_tmp = np.zeros((1440, 64, 12))

for i in range(0, 1440):
    for j in range(0, 64):
        frame_prob_tmp[i, j, np.argmax(predictions[i, j, :])] = 1

frames_prob_sum = []
mat_percentage_per_frame = np.zeros((12, 12))
count = 0

for i in range(0, 12):
    for j in range(0, 12):
        for k in range(0, 10):
            mat_percentage_per_frame[j, :] += np.sum(frame_prob_tmp[count, :, :], axis=0)
            count += 1

    
mat_percentage_per_frame  = mat_percentage_per_frame /(64*10*12)
print('The average per-frame accuracy derived from confusion matrix is:\n')
print(np.trace(mat_percentage_per_frame)/12)

#============================================
frame_prob_tmp = np.zeros((1440, 64, 12))

for i in range(0, 1440):
    for j in range(0, 64):
        frame_prob_tmp[i, j, np.argmax(predictions[i, j, :])] = 1

frames_prob_sum = []
mat_percentage_seq = np.zeros((12, 12))
count = 0

for i in range(0, 12):
    for j in range(0, 12):
        for k in range(0, 10):
            file_seq = np.sum(frame_prob_tmp[count, :, :], axis=0)
            mat_percentage_seq[j, np.argmax(file_seq)] += 1
            count += 1

    
mat_percentage_seq = mat_percentage_seq/(12*10)
print('The average sequence accuracy derived from confusion matrix is:\n')
print(np.trace(mat_percentage_seq)/12)

output_dir = '/home/lab210/MattYen_workspace/gest_August/cross-scene/NEW/lstm/3t4rRDI/30-70/'
#output_dir = h5_dirNEW/lstm/3t4rRDIz

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
#fig_loss.savefig(output_dir + 'loss.png')



model.save(output_dir + 'model_NewConfig_bilstm_3t4rRAI_32x32_Gaussian_batch12_ProgressSGD.h5')
model.save_weights(output_dir + 'weightsNewConfig_bilstm_3t4rRAI_32x32_Gaussian_batch12_ProgressSGD.h5')

