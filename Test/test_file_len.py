# from camera_capture_mp4 import CamCapture,VideoWriterWidget
from queue import Queue
import threading
import  cv2
import  time
import numpy as np

path = 'C:/Users/user/Desktop/thmouse_training_data/'

x = np.load(path+"raw.npy")
print(x.shape)