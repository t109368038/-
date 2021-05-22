import time
from time import time as timer
from threading import Thread
from PyQt5.QtCore import QThread
import keyboard  # using module keyboard

import cv2


class RTSPVideoWriterObject(QThread):

    def __init__(self, src=0, filename="",save_frame_len=0):
        super(RTSPVideoWriterObject, self).__init__()
        # Create a VideoCapture object
        self.wc_number = src
        self.record_frame_len  = save_frame_len
        self.capture = cv2.VideoCapture(src)
        self.capture.set(3, 640)  # 設定解析度
        self.capture.set(4, 480)
        self.record = False
        self.counter = 0
        # Default resolutions of the frame are obtained (system dependent)
        self.frame_width = int(self.capture.get(3))
        self.frame_height = int(self.capture.get(4))

        # Set up codec and output video settings
        self.codec = cv2.VideoWriter_fourcc(*'mp4v')
        mp4_path = 'C:/Users/user/Desktop/thmouse_training_data/'+ filename +".mp4"
        self.output_video = cv2.VideoWriter(mp4_path, self.codec, 30, (self.frame_width, self.frame_height))
        self.readframe = 0
        self.fps_count = 0
        self.start_time = time.time()
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        print("--- cam{} init work ---".format(src))

        # Start the thread to read frames from the video stream

    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
                if self.record:
                    self.save_frame()
                    self.readframe += 1
                self.fps_count += 1

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

    def close_webcam(self):
        #close the webcam process
        print("closewebcam")
        self.end = time.time()
        self.capture.release()
        self.output_video.release()
        cv2.destroyAllWindows()
        # exit(1)

    def get_fsp(self):
        if self.fps_count != 0 :
            fps =  self.fps_count/(time.time() - self.start_time)
            print("Time cam{} FPS is  : {} frame/perseconds".format(self.wc_number,fps))



if __name__ == '__main__':
    rtsp_stream_link = 0
    rtsp_stream_link1 = 1
    save_frame_len = 120
    video_stream_widget = RTSPVideoWriterObject(rtsp_stream_link, "vedio1",save_frame_len=save_frame_len)
    video_stream_widget2 = RTSPVideoWriterObject(rtsp_stream_link1, "vedio2",save_frame_len=save_frame_len)

    print("start")
    while(1):
        print(video_stream_widget2.counter)
        if keyboard.is_pressed('s'):  # if key 'q' is pressed
            video_stream_widget.record = True
            video_stream_widget2.record = True
        if video_stream_widget.counter >= 120:
            video_stream_widget.close_webcam()
            video_stream_widget.record = False
        if video_stream_widget2.counter >= 120:
            video_stream_widget2.close_webcam()
            video_stream_widget2.record = False
        if  not(video_stream_widget.record & video_stream_widget2.records):
            break
