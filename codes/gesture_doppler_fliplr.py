import numpy as np


vali_dir='/home/pstudent/DeepSoli Models/gesture_data/Range-Doppler_fair comparison/3t4r_H104_NewConfig/'

x_test = np.load(vali_dir + 'x_test.npy')
y_test = np.load(vali_dir + 'y_test.npy') 
x_test = np.transpose(x_test, (0,1,3,4,2))

x_fliped = np.zeros((1010, 64, 32, 64, 12))

for i in range(0, 1010):
    print(i)
    for j in range(0, 12):
        for k in range(0, 64):
            x_fliped[i,k,:,:,j] = np.fliplr(x_test[i,k,:,:,j])
        