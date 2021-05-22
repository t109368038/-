import time
from time import time as timer

from threading import Thread
import keyboard  # using module keyboard
import cv2

class RTSPVideoWriterObject(object):
    def __init__(self, src=0, filename=""):
        # Create a VideoCapture object
        self.number = src
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

        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        self.readframe = 0

    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()

                if (self.record) :
                    self.save_frame()

                if self.record:
                    self.readframe += 1

    def show_frame(self):
        # Display frames in main program
        self.start = timer()
        if self.status:
            cv2.imshow('frame'+str(self.number), self.frame)

        # Press Q on keyboard to stop recording
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.capture.release()
            self.output_video.release()
            cv2.destroyAllWindows()
            exit(1)
        # Press spacebar to start/stop recording
        elif key == 32:
            if self.record:
                self.record = False
                print('Stop recording')
            else:
                self.record = True
                print('Start recording')

    def save_frame(self):
        # Save obtained frame into video output file
        self.output_video.write(self.frame)
        self.counter += 1


if __name__ == '__main__':
    rtsp_stream_link = 0
    rtsp_stream_link1 = 1
    frame_len = 120
    video_stream_widget = RTSPVideoWriterObject(rtsp_stream_link, "vedio1")
    video_stream_widget2 = RTSPVideoWriterObject(rtsp_stream_link1, "vedio2")
    while True:

        try:
            video_stream_widget.show_frame()
            video_stream_widget2.show_frame()
            print(video_stream_widget.counter)
            print(video_stream_widget2.counter)
            # print(video_stream_widget2.readframe)
            if keyboard.is_pressed('s'):  # if key 'q' is pressed
                video_stream_widget.record = True
                # start1 = timer()
                start1 = time.time()

                video_stream_widget2.record = True
                # start2 = timer()
                start2 = time.time()

            # if video_stream_widget.counter >=120 & video_stream_widget2.counter >=120:
            if video_stream_widget.counter >= frame_len and video_stream_widget2.counter>=frame_len :
                video_stream_widget.record  = False
                # end1 = timer()
                end1 = time.time()
                video_stream_widget2.record  = False
                # end2 = timer()
                end2 = time.time()

                video_stream_widget.capture.release()
                video_stream_widget.output_video.release()
                # print(video_stream_widget.readframe)
                video_stream_widget2.capture.release()
                video_stream_widget2.output_video.release()
                cv2.destroyAllWindows()
                # exit(1)
                break

        except AttributeError:
            pass

    seconds1 = end1 - start1
    seconds2 = end2 - start2
    print("Time cam1 taken : {0} seconds".format(seconds1))
    print("Time cam2 tsaken : {0} seconds".format(seconds2))
    fps1 = frame_len / seconds1
    fps2 = frame_len / seconds2
    print("Estimated cam1's frames per second : {0}".format(fps1))
    print("Estimated cam1's frames per second : {0}".format(fps2))


