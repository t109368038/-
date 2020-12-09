from __future__ import print_function
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dropout, Flatten, Dense, LSTM, TimeDistributed, Bidirectional
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Multiply, Permute, Reshape, Input, RepeatVector
from tensorflow.keras.models import load_model, Model
import tensorflow.keras.optimizers as optimizers
from sklearn import metrics
import numpy as np
#import tensorflow.config as config
from tensorflow.keras.utils import multi_gpu_model, to_categorical
from sklearn.metrics import confusion_matrix
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn import manifold

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"


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


#h5_dir='/home/pstudent/DeepSoli Models/gesture_data/Range-Doppler_fair comparison/3t4r_lab210_cropped_whole as a set/'
##model_dir = '/home/pstudent/DeepSoli Models/gesture_data/Range-Doppler_fair comparison/all as trainset_3t4r/'
#vali_dir='/home/pstudent/DeepSoli Models/gesture_data/Range-Doppler_fair comparison/3t4r_H104_cropped_whole as a set/'
h5_dir='/home/pstudent/DeepSoli Models/gesture_data/Range-Doppler_fair comparison/3t4r_lab210_cropped_whole as a set/'
#model_dir = '/home/pstudent/DeepSoli Models/gesture_data/Range-Doppler_fair comparison/3t4r_lab210_cropped_whole as a set/vali with 104/[original NN]/'
vali_dir='/home/pstudent/DeepSoli Models/gesture_data/Range-Doppler_fair comparison/3t4r_H104_cropped_whole as a set/'

x_train = np.load(h5_dir + 'x_train.npy')
y_train = np.load(h5_dir + 'y_train.npy') 
print (x_train.shape) 
print (y_train.shape) 

#x_test = np.load(h5_dir + 'x_test.npy')
#y_test = np.load(h5_dir + 'y_test.npy') 
#print (x_test.shape)
#print (y_test.shape) 

x_train = np.transpose(x_train, (0,1,3,4,2))
#x_test = np.transpose(x_test, (0,1,3,4,2))


x_test = np.load(vali_dir + 'x_test.npy')
y_test = np.load(vali_dir + 'y_test.npy') 
x_test = np.transpose(x_test, (0,1,3,4,2))

x_train = x_train[:,:,:,:,1]
x_test = x_test[:,:,:,:,1]
x_train = np.reshape(x_train, (1340, 64, 1024))
x_test = np.reshape(x_test, (1010, 64, 1024))





    

#y_train = to_categorical(y_train, 12)
#y_test = to_categorical(y_test, 12)

#x_train = np.reshape(x_train, (1340*12, 64, 1024))
#x_test = np.reshape(x_test, (1010*12, 64, 1024))
#y_train = np.reshape(y_train, (1340*12, 64))
#y_test = np.reshape(y_test, (1010*12, 64))

#x_train=x_train[:,:,:,:,:]
#x_test=x_test[:,:,:,:,:]

#x_train = x_train[..., np.newaxis]
#x_test = x_test[..., np.newaxis]

#x_train = (x_train > -0.40) * x_train
#x_test = (x_test > -0.40) * x_test

#==============================================================================
#model = load_model(model_dir + 'model_bilstm_RangeDoppler_32x32_255normalized_batch12_ProgressSgd.h5')
#model.load_weights(model_dir + 'weights_bilstm_RangeDoppler_32x32_255normalized_batch12_ProgressSgd.h5', by_name = True)
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

 
scheduler = LearningRateScheduler(lr_scheduler)
# =============================================================================

    
# =============================================================================
#x_train = x_train.astype('float32') / 255. - 0.5
#x_test = x_test.astype('float32') / 255. - 0.5
# =============================================================================
with tf.device('/gpu:0'):
    
    model = Sequential()
    model.add(TimeDistributed(Dense(1024), input_shape=x_train.shape[1:]))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Dense(512)))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Dense(256)))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Dense(128)))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    
    model.add(TimeDistributed(Dense(64)))
    
    model.add(TimeDistributed(Dense(128)))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Dense(256)))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Dense(512)))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Dense(1024)))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(Activation('relu')))
#    model.add(TimeDistributed(Conv2D(128, (3, 3)), input_shape=x_train.shape[1:]))
#    #model.add(TimeDistributed(MaxPooling2D((3, 3), strides=1, padding='valid', data_format='channels_last')))
#    model.add(TimeDistributed(BatchNormalization()))
#    model.add(TimeDistributed(Activation('relu')))
#    model.add(TimeDistributed(Dropout(0.4)))
    
#    model.add(TimeDistributed(Dense(128)))
#    model.add(TimeDistributed(BatchNormalization()))
#    model.add(TimeDistributed(Activation('relu')))
#    model.add(TimeDistributed(Dropout(0.4)))
#    model.add(TimeDistributed(Flatten(), input_shape=[64, 32, 32, 12]))
#    model.add(Reshape((64, 1024), input_shape=x_train.shape[1:]))
#    model.add(LSTM(1024, activation='relu', return_sequences=False))
#    model.add(RepeatVector(64))
#    model.add(LSTM(1024, return_sequences=True))
#    model.add(TimeDistributed(Dense(12)))
#    model.add(TimeDistributed(Activation('softmax')))
#==============================================================================
#     x = TimeDistributed(Conv2D(32, (3, 3), strides=1, padding='valid'), input_shape=x_train.shape[1:])(input_seq)
#     #TimeDistributed(MaxPooling2D((3, 3), strides=1, padding='valid', data_format='channels_last')))
#     x = TimeDistributed(BatchNormalization())(x)
#     x = TimeDistributed(Activation('relu'))(x)
# #    TimeDistributed(Dropout(0.4)))
#     
#     x = TimeDistributed(Conv2D(64, (3, 3)))(x)
#     #TimeDistributed(MaxPooling2D((3, 3), strides=1, padding='valid', data_format='channels_last')))
#     x = TimeDistributed(BatchNormalization())(x)
#     x = TimeDistributed(Activation('relu'))(x)
#     x = TimeDistributed(Dropout(0.4))(x)
#     
#     x = TimeDistributed(Conv2D(128, (3, 3)))(x)
#     #TimeDistributed(MaxPooling2D((3, 3), strides=1, padding='valid', data_format='channels_last')))
#     x = TimeDistributed(BatchNormalization())(x)
#     x = TimeDistributed(Activation('relu'))(x)
#     x = TimeDistributed(Dropout(0.4))(x)
#     
#     x = TimeDistributed(Flatten())(x)
#     
#     x = TimeDistributed(Dense(512))(x)
#     x = TimeDistributed(BatchNormalization())(x)
#     x = TimeDistributed(Activation('relu'))(x)
#     x = TimeDistributed(Dropout(0.5))(x)
#     
#     x = TimeDistributed(Dense(512))(x)
#     x = TimeDistributed(BatchNormalization())(x)
#     x = TimeDistributed(Activation('relu'))(x)
#     x = TimeDistributed(Dropout(0.5))(x)
# #    model.add(TimeDistributed(Dropout(0.5)))
#     #model.add(TimeDistributed(Lambda(lambda x: tf.expand_dims(model.output, axis=-1))))
# #    model.add(Bidirectional(LSTM(512, return_sequences=True)))
# #    Att_Output = Bidirectional(GRU(512, return_sequences=True))(x)
#     Att_Input = Bidirectional(LSTM(512, return_sequences=True))(x)    
#     Attention = Permute([2, 1])(Att_Input)
#     Attention = TimeDistributed(Dense(64, activation='softmax'))(Attention)
#     Probs = Permute([2, 1])(Attention)
#     Att_Output = Multiply()([Att_Input, Probs])
#     
#     Att_Output = TimeDistributed(Flatten())(Att_Output)
#    Att_Output = TimeDistributed(Dropout(0.5))(Att_Output)
#    x = TimeDistributed(Dense(12))(Att_Output)
#    pred = TimeDistributed(Activation('softmax'))(x)
#==============================================================================

########################################################################################################
#model.save(h5_dir + 'model_uni-lstm_RangeAngle_64x256_255normalized_12label_80-20_batch12_ProgressSgd.h5')
#model.save_weights(h5_dir + 'weights_uni-lstm_RangeAngle_64x256_255normalized_12label_80-20_batch12_ProgressSgd.h5')

#model = Model(inputs=input_seq, outputs=pred)

sgd = optimizers.SGD(lr=0.0, momentum=0.9, decay=0.0, nesterov=False)
model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

history = model.fit(x_train, y_train, 
          batch_size=12, epochs=40, shuffle=True, verbose=1, 
          callbacks=[scheduler],
          validation_data = (x_test, y_test)
          )

#score = model.evaluate(x_test, y_test, batch_size=12)

#history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=7),
#                              epochs=60, shuffle=True, verbose=1, 
##          callbacks=[scheduler],
#          validation_data = (x_test, y_test))

#model.fit(x_train, y_train, batch_size=7, epochs=40, shuffle=True, verbose=1)

predictions = model.predict(x_test)
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

frames_prob_sum=[]
temp=[]
mat_percentage = np.zeros((12, 12))

for i in range(0, 64):
    frames_prob_sum = np.sum(predictions, axis=1) # sum as (test_num, label)

count=0;
for j in range(0, 7):      
    temp = sum(frames_prob_sum[count:count+90,:])
    mat_percentage[j, :] = temp/90
    count+=90       

for j in range(7, 10):
    temp = sum(frames_prob_sum[count:count+60,:])
    mat_percentage[j, :] = temp/60
    count+=60

for j in range(10, 12):
    temp = sum(frames_prob_sum[count:count+100,:])
    mat_percentage[j, :] = temp/100
    count+=100
    
mat_percentage = mat_percentage/64
print('The vali accuracy derived from confusion matrix is:\n')
print(np.trace(mat_percentage)/12)
#==============================================================================

########################################################
#tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
#X_tsne = tsne.fit_transform(xx)
#
#x_min, x_max = X_tsne.min(0), X_tsne.max(0)
#X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
#plt.figure(figsize=(8, 8))
#for i in range(X_norm.shape[0]):
#    plt.text(X_norm[i, 0], X_norm[i, 1], str(yy[i]), color=plt.cm.Set1(yy[i]), 
#             fontdict={'weight': 'bold', 'size': 9})
#plt.xticks([])
#plt.yticks([])
#plt.show()

#
# summarize history for accuracy
fig_acc = plt.gcf()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.draw()


# summarize history for loss
fig_loss = plt.gcf()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.draw()

#fig_acc.savefig(h5_dir +'acc.jpg')
#fig_loss.savefig(h5_dir + 'loss.png')

#model.save(h5_dir + 'model_bilstm_RangeDoppler_32x32_255normalized_batch12_ProgressSgd.h5')
#model.save_weights(h5_dir + 'weights_bilstm_RangeDoppler_32x32_255normalized_batch12_ProgressSgd.h5')

#model = load_model('/home/pstudent/DeepSoli Models/gesture_data/Range-Angle_fair comparison/1t4r/H104/lstm/model_H104_1t4r_lstm_RangeAngle_64x64_255normalized_12label_80-20_ProgressSgd.h5')