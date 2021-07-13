import numpy as np

path = 'D:\\ForAndy\\1443_point_cloud\\'
x = np.load(path+"point_cloud0_0_0_001"+".npy", allow_pickle=True)

print(x[8].shape)
print("ok")