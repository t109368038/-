import cv2
import threading as th
import time
import mediapipe as mp
import sys


class CamCapture(th.Thread):
    def __init__(self, thread_id, name, counter, th_lock, mode=0):
        th.Thread.__init__(self)
        self.threadID = thread_id
        self.name = name
        self.counter = counter
        self.fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
        self.frame = 0
        self.lock = th_lock
        self.mode = mode

    def run(self):
        if self.mode == 0:
            # mediapipe
            mp_drawing = mp.solutions.drawing_utils
            mp_hands = mp.solutions.hands
            hands = mp_hands.Hands(
                min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=1)
            cam = cv2.VideoCapture(self.counter)
            print("%s: %s\n" % (self.name, time.ctime(time.time())))
            while cam.isOpened():
                ret, self.frame = cam.read()
                if not ret:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    continue
                image = cv2.cvtColor(cv2.flip(self.frame, 1), cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = hands.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                cv2.imshow('MediaPipe Hands' + str(self.counter), image)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
            cv2.destroyWindow('MediaPipe Hands' + str(self.counter))
            cam.release()
            hands.close()
            print('Close process')
            print("%s: %s" % (self.name, time.ctime(time.time())))
        elif self.mode == 1:
            # no mediapipe
            cv2.namedWindow(self.name)
            cam = cv2.VideoCapture(self.counter)
            ret, self.frame = cam.read()
            cv2.imshow(self.name, self.frame)
            print("%s: %s\n" % (self.name, time.ctime(time.time())))
            while cam.isOpened():
                ret, self.frame = cam.read()
                cv2.imshow(self.name, self.frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cv2.destroyWindow(self.name)
            cam.release()
            cv2.VideoCapture(0)
            print('Close process')
            print("%s: %s" % (self.name, time.ctime(time.time())))
        else:
            raise ValueError('CamCapture does not have this mode.')


lock = th.Lock()
cam1 = CamCapture(0, 'First', 0, lock, 3)
# cam2 = CamCapture(1, 'Second', 1, lock)
cam1.start()
cam1.join()
# cam2.start()
# threads = [cam1, cam2]
# for t in threads:
#     t.join()
# sys.exit(0)
