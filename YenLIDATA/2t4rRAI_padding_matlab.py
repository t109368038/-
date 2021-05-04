import h5py
import matplotlib.pyplot as plt
import numpy as np
import sys

path1 = 'C:/data2/padding_h5_2t4r/RAI/'
path2 = 'C:/data2/padding_h5_2t4r/RAI_rotate/'

# file = h5py.File(path + '3t4r_0_0_0_001.h5', 'a')

# data = file['data']
# plt.figure()
# plt.imshow(np.rot90(data[0, 0, :, :], -1))

scene = [0, 1]
gesture = 12
person = 12
times = 10

for s in scene:
    for g in range(gesture):
        gesture_tmp = []
        for p in range(person):
            for t in range(times):
                name = '3t4r_' + str(s) + '_' + str(p) + '_' + str(g) + '_00' + str(t + 1) + '.h5'
                file = h5py.File(path1 + name)
                data = file['data']
                data_tmp = np.zeros([64, 1, 32, 32])
                for f in range(64):
                    data_tmp[f, 0] = np.rot90(data[f, 0], -1)
                gesture_tmp.append(data_tmp)
        print(np.shape(gesture_tmp))
        np.save(path2 + 'S' + str(s) +'G' + str(g), gesture_tmp)

sys.exit(0)