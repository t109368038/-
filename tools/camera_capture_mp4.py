import cv2
import threading as th
import time
import mediapipe as mp
import sys
from PyQt5.QtCore import QThread, pyqtSignal


class CamCapture(th.Thread):
    def __init__(self, thread_id, name, counter, th_lock, cam_queue=None, save_queue=None, status=0, mode=0,mp4_path=''):
        th.Thread.__init__(self)
        self.threadID = thread_id
        self.name = name
        self.counter = counter
        self.fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
        self.lock = th_lock
        self.mode = mode
        self.cam_queue = cam_queue
        self.save_queue = save_queue
        self.status = status
        self.save_mp4_path = mp4_path
        print('Camera Capture Mode:{}'.format(mode))
        print('========================================')
        if mode ==0 :
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=1)

    def run(self):
        self.cam = cv2.VideoCapture(self.counter)
        # self.cam.set(3, 1920)  # 設定解析度
        # self.cam.set(4, 1080)
        self.cc= 0

        self.frame_count = 0
        sz = (int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.vout = cv2.VideoWriter()
        self.vout.open(self.save_mp4_path + 'output'+str(self.counter)+'.mp4', self.fourcc, 30, sz, True)
        fps = int(self.cam.get(5))
        print('FPS:{}'.format(fps))
        ret, frame = self.cam.read()
        tmp_frame = frame
        print('Camera is opened')
        print("Camera[%s] open time: %s" % (self.counter, time.ctime(time.time())))
        print('========================================')
        start = time.time()
        while self.cam.isOpened():
            ret, frame = self.cam.read()
            if self.mode == 0:
                image = cv2.cvtColor(cv2.flip(frame.copy(), 1), cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = self.hands.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                cv2.imshow(self.name, image)
            else:
                cv2.imshow(self.name, frame)

            if self.status ==1:
                self.vout.write(frame)
                self.cc += 1
                if self.cc >=120:
                    break
            cv2.waitKey(1)
        end = time.time()
        seconds = end - start
        print("Time taken : {0} seconds".format(seconds))
        fps = 120 / seconds;
        print("Estimated frames per second : {0}".format(fps))
        print("len is {}".format(self.cc))
        cv2.destroyWindow(self.name)
        self.cam.release()
        # self.vout.release()
        print('Close process')
        print("%s: %s" % (self.name, time.ctime(time.time())))

    def close(self):
        self.cam.release()
        # self.vout.release()
