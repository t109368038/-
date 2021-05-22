from camera_capture_mp4 import CamCapture,VideoWriterWidget
from queue import Queue
import threading
import  cv2
import  time
CAMData = Queue()
CAMData2 = Queue()
cam_rawData = Queue()
cam_rawData2 = Queue()

lock = None
# cam1 = CamCapture(0, 'First', 0, lock, CAMData, cam_rawData, mode=1,
#                   mp4_path='C:/Users/user/Desktop/thmouse_training_data/')
# cam2 = CamCapture(1, 'Second', 1, lock, CAMData2, cam_rawData2, mode=1,
#                   mp4_path='C:/Users/user/Desktop/thmouse_training_data/')

mp4_path='C:/Users/user/Desktop/thmouse_training_data/'
cam0  = VideoWriterWidget(Save_Path=mp4_path,video_file_name="outvedio_cam0",src=0)
cam1  = VideoWriterWidget(Save_Path=mp4_path,video_file_name="outvedio_cam1",src=1)

cam0.start()
cam1.start()



# cam.set(3, 1080)  # 設定解析度
# cam.set(4, 960