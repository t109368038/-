import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import keyboard
import numpy as np
import mediapipe as mp
from collections import deque
import argparse
import math
from typing import List, Tuple, Union
import matplotlib.pyplot as plt
import sys
from mediapipe_proecss import _normalized_to_pixel_coordinates,larry_draw,plot_hand
from mediapipe_proecss import  plot_hand

def print_test_3d():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(0, 0, 0, marker="o")
    line=[]
    x = np.load("saveline_x.npy")
    y = np.load("saveline_y.npy")
    z = np.load("saveline_z.npy")
    plot_hand(fig, -135, 69, x, y, z,[],[],[], "aavg")

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    # static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    )



# data_path = 'C:/Users/user/Desktop/2021-0317/'
# data_path = 'C:/Users/user/Desktop/2021-03-31'
# data_path = 'D:/kaiku_report/20210414'
data_path = 'E:/NTUT-master/KaiKu Report/0426/'

videoCapture = cv2.VideoCapture(data_path+"output0.mp4")
videoCapture1 = cv2.VideoCapture(data_path+"output1.mp4")

fps = videoCapture.get(cv2.CAP_PROP_FPS)
size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fNUMS = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)

fps1 = videoCapture1.get(cv2.CAP_PROP_FPS)
size1 = (int(videoCapture1.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(videoCapture1.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fNUMS1 = videoCapture1.get(cv2.CAP_PROP_FRAME_COUNT)

red = [0, 0, 255]
red_color = (0, 0, 255) # BGR
cc = 0
# fig1 = plt.figure()
# fig1.show()
runrun=0
line = np.array([])
x=[]
y=[]
z=[]
loose_time = 0
is_pass = False
only_one = 1
frame_lose = 0
frame1_lose = 0
text = None
text1 = None
cam_hp=[]
cam1_hp=[]
success, frame = videoCapture.read()
success1, frame1 = videoCapture1.read()
#----------------------
sz = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
vout = cv2.VideoWriter()
vout.open(data_path + '/video'+str(only_one)+ '.mp4',fourcc, 20, sz, True)


frames_num=videoCapture.get(7)
print(frames_num)
count = 0
#------------------------------------------
while success:
    count += 1
    frame.flags.writeable = False
    frame1.flags.writeable = False
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)

    frame = cv2.flip(frame, 1)
    frame = cv2.flip(frame, 0)
    frame1 = cv2.flip(frame1, 1)
    frame1 = cv2.flip(frame1, 0)


    if only_one == 0:
        results = hands.process(frame)
        if results.multi_hand_landmarks :
            for hand_landmarks in results.multi_hand_landmarks:
                x1, y1 = larry_draw(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cam_hp = cam_hp + [x1, y1]
        else:
            frame_lose +=1
            text = "the frame lose_times: " + str(frame_lose)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # frame = np.flip(frame)
        frame = np.rot90(frame, 2)
        # if text != None:
        #     cv2.putText(frame, text, (10, 430), cv2.FONT_HERSHEY_COMPLEX_SMALL,
        #                 1, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("frame",frame)
        vout.write(frame)

    elif only_one == 1:
        results1 = hands.process(frame1)
        if results1.multi_hand_landmarks :
            for hand_landmarks in results1.multi_hand_landmarks:
                x2, y2 = larry_draw(frame1, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(
                    frame1, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cam1_hp = cam1_hp + [x2, y2]
        else:
            cam1_hp = cam1_hp + [[np.zeros(21)], np.zeros[21]]
            frame1_lose +=1
            text1 = "the frame lose_times: " + str(frame1_lose)
        # if text1 != None:
        #     cv2.putText(frame1, text1, (10, 430), cv2.FONT_HERSHEY_COMPLEX_SMALL,
        #                 1, (0, 255, 255), 1, cv2.LINE_AA)
        frame1 = np.flip(frame1,2)
        frame1 = np.rot90(frame1,2)
        cv2.imshow("frame",frame1)
        vout.write(frame1)

    else:
        results = hands.process(frame)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x1, y1 = larry_draw(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        else:
            frame_lose += 1
            text = "the frame lose_times: " + str(frame_lose)
        if text != None:
            cv2.putText(frame, text, (10, 430), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        1, (0, 255, 255), 1, cv2.LINE_AA)
        if text1 != None:
            cv2.putText(frame, text1, (10, 460), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        1, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "Total frames:"+str(fNUMS) , (420, 460), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    1, (0, 255, 255), 1, cv2.LINE_AA)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        cv2.imshow("frame", frame)
        results1 = hands.process(frame1)

        if results1.multi_hand_landmarks :
            for hand_landmarks in results1.multi_hand_landmarks:
                x2, y2 = larry_draw(frame1, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(
                    frame1, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        else:
            frame1_lose += 1
            text1 = "the frame1 lose_times: " + str(frame1_lose)
        frame1 = np.rot90(frame1, -1)
        cv2.imshow("frame1",frame1)


    success, frame = videoCapture.read()
    success1, frame1 = videoCapture1.read()

    cv2.waitKey(30) #延迟

if only_one ==0:
    videoCapture.release()
    np.save(data_path + "/cam_hp.npy",cam_hp)
    vout.release()

elif only_one == 1:
    videoCapture1.release()
    np.save(data_path + "/cam_hp1.npy",cam1_hp)
    vout.release()
