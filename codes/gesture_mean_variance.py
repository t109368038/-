import numpy as np

h5_dir='/home/pstudent/DeepSoli Models/gesture_data/Range-Angle_fair comparison/3t4r/lab210_NewConfig/'
#model_dir = '/home/pstudent/DeepSoli Models/gesture_data/Range-Doppler_fair comparison/all as trainset_3t4r/'
vali_dir='/home/pstudent/DeepSoli Models/gesture_data/Range-Angle_fair comparison/3t4r/H104_NewConfig/'
#h5_dir = '/home/pstudent/DeepSoli Models/gesture_data/Range-Angle_fair comparison/3t4r/scene_mixed/'
#h5_dir = '/home/pstudent/DeepSoli Models/gesture_data/transformto_h5/'

x_train = np.load(h5_dir + 'x_train.npy')
y_train = np.load(h5_dir + 'y_train.npy') 

#x_test = np.load(h5_dir + 'x_test.npy')
#y_test = np.load(h5_dir + 'y_test.npy') 
#print (x_test.shape)
#print (y_test.shape) 

x_train = np.transpose(x_train, (0,1,3,4,2))
#x_test = np.transpose(x_test, (0,1,3,4,2))

#x_test = np.transpose(x_test, (0,1,3,4,2))
x_train = x_train*255

#==========================================
#x_train = x_train[:,:,:,32:,:]
#==========================================

num_train = x_train.shape[0]
width_train = x_train.shape[2]
height_train = x_train.shape[3]
channel_train = x_train.shape[4]


x_train_mean_perChannel=np.zeros((num_train,64,width_train,height_train,channel_train))
for i in range(0, channel_train):
    temp = np.mean(x_train[:,:,:,:,i])
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
    x_train_std[:,j] = np.std(x_train[:,:,:,:,j])

x_train_meanstd = (x_train - x_train_mean_perChannel)/x_train_std

np.save('x_train_meanstd.npy', x_train_meanstd)

#===================================================================================++#

x_test = np.load(vali_dir + 'x_test.npy')
y_test = np.load(vali_dir + 'y_test.npy') 
x_test = np.transpose(x_test, (0,1,3,4,2))

x_test = x_test*255

#===================================
#x_test = x_test[:,:,:,32:,:]
#===================================

num_test = x_test.shape[0]
width_test= x_test.shape[2]
height_test = x_test.shape[3]
channel_test = x_test.shape[4]

x_test_mean_perChannel=np.zeros((num_test,64,width_test,height_test,channel_test))
for i in range(0, channel_test):
    temp = np.mean(x_test[:,:,:,:,i])
    temp = np.repeat(temp, height_test, axis=0)
    temp_2 = temp[np.newaxis,...]
    temp_2 = np.repeat(temp_2, width_test, axis=0)
    temp_2 = temp_2[np.newaxis,...]
    temp_2 = np.repeat(temp_2, 64, axis=0)
    temp_3 = temp_2[np.newaxis,...]
    temp_3 = np.repeat(temp_3, num_test, axis=0)
    x_test_mean_perChannel[:,:,:,:,i] = temp_3
    print(i)


    
x_test_std = np.zeros((1, channel_test))
for j in range(0, channel_test):
    x_test_std[:,j] = np.std(x_test[:,:,:,:,j])
    
x_test_meanstd = (x_test - x_test_mean_perChannel)/x_test_std

np.save('x_test_meanstd.npy', x_test_meanstd)