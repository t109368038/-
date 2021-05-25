import time
from time import time as timer
from threading import Thread
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import keyboard  # using module keyboard
import numpy as np
import cv2
from PyQt5.QtGui import QPixmap
from qimage2ndarray.dynqt import QtGui


class RTSPVideoWriterObject(QThread):
    change_pixmap_signal = pyqtSignal(QPixmap)

    def __init__(self, src=0, filename="",save_frame_len=0):
        super(RTSPVideoWriterObject, self).__init__()
        # Create a VideoCapture object
        self.wc_number = src
        self.record_frame_len  = save_frame_len
        self.capture = cv2.VideoCapture(src)
        # self.capture.set(cv2.CV_CAP_PROP_FPS, 25)
        self.capture.set(cv2.CAP_PROP_FPS, 20)
        print(self.capture.get(cv2.CAP_PROP_FPS))

        self.capture.set(3, 640)  # 設定解析度
        self.capture.set(4, 480)
        self.record = False
        self.counter = 0
        # Default resolutions of the frame are obtained (system dependent)
        self.frame_width = int(self.capture.get(3))
        self.frame_height = int(self.capture.get(4))
        # Set up codec and output video settings
        self.codec = cv2.VideoWriter_fourcc(*'mp4v')
        self.mp4_path = 'C:/Users/user/Desktop/thmouse_training_data/'+ filename +".mp4"
        self.output_video = cv2.VideoWriter(self.mp4_path, self.codec, 20, (self.frame_width, self.frame_height))
        self.readframe = 0
        self.fps_count = 0
        self.start_time = time.time()
        print("--- cam{} init work ---".format(self.wc_number))

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line,
                                            QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(cv_img.shape[1], cv_img.shape[0], Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def run(self):
        # self.thread = Thread(target=self.update, args=())
        # self.thread.daemon = True
        # self.thread.start()
        # self.thread.join(timeout=1)
        self.update()
        # Start the thread to read frames from the video stream

    def update(self):
        # Read the next frame from the stream in a different thread
        timestart = time.time()
        while True:
            if self.capture.isOpened():

                (self.status, self.frame) = self.capture.read()

                # print("cam1{} gogo ".format(self.wc_number))
                if self.record:
                    img = self.convert_cv_qt(self.frame)
                    self.change_pixmap_signal.emit(img)
                    print("cam{} frame:{}".format(self.wc_number,self.readframe))
                    self.save_frame()
                    self.readframe += 1

                time_end = time.time() - timestart
                self.fps_count += 1
                fps = self.fps_count/time_end
                if self.fps_count % 100 ==0:
                    print(fps)

    def show_frame(self):
        # Display frames in main program
        if self.status:
            cv2.imshow('frame'+str(self.wc_number), self.frame)

    def save_frame(self):
        # Save obtained frame into video output file
        self.output_video.write(self.frame)
        self.counter += 1

    def state_change(self):
        self.record = not self.record

    def release_video(self):
        #close the webcam process
        self.output_video.release()
        # self.output_video = cv2.VideoWriter(self.mp4_path, self.codec, 30, (self.frame_width, self.frame_height))

    def get_fsp(self):
        if self.fps_count != 0 :
            fps =  self.fps_count/(time.time() - self.start_time)
            print("Time cam{} FPS is  : {} frame/perseconds".format(self.wc_number,fps))

    def restart_videowriter(self):
        self.output_video = cv2.VideoWriter(self.mp4_path, self.codec, 30, (self.frame_width, self.frame_height))
        self.readframe = 0

if __name__ == '__main__':
    rtsp_stream_link = 0
    rtsp_stream_link1 = 1
    save_frame_len = 120
    # video_stream_widget = RTSPVideoWriterObject(rtsp_stream_link, "vedio1",save_frame_len=save_frame_len)
    # video_stream_widget2 = RTSPVideoWriterObject(rtsp_stream_link1, "vedio2",save_frame_len=save_frame_len)
    # print("start")
    # while(1):
    #     print(video_stream_widget2.counter)
    #     if keyboard.is_pressed('s'):  # if key 'q' is pressed
    #         video_stream_widget.record = True
    #         video_stream_widget2.record = True
    #     if video_stream_widget.counter >= 120:
    #         video_stream_widget.close_webcam()
    #         print(type(video_stream_widget2.frame))
    #
    #         video_stream_widget.record = False
    #     if video_stream_widget2.counter >= 120:
    #         video_stream_widget2.close_webcam()
    #         video_stream_widget2.record = False
    #     if  not(video_stream_widget.record & video_stream_widget2.record):
    #         break
    def update_image(img):
        cv2.imshow("mats",img)

    video_stream_widget = RTSPVideoWriterObject(rtsp_stream_link, "vedio1",save_frame_len=save_frame_len)
    video_stream_widget.start()
    video_stream_widget.change_pixmap_signal.connect(update_image)
    video_stream_widget.record = True

    # while True:
    #     if keyboard.is_pressed('s'):  # if key 'q' is pressed
    #         video_stream_widget.record = False
    #         video_stream_widget.close_webcam()
    #         break
