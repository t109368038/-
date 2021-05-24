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
data_path = 'C:/Users/user/Desktop/thmouse_training_data/'

# =================
videoCapture  =  cv2.VideoCapture(data_path+"vedio1.mp4")
fps = videoCapture.get(cv2.CAP_PROP_FPS)
size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fNUMS = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
# =================
only_one = 0
# =================
red = [0, 0, 255]
red_color = (0, 0, 255) # BGR
cc = 0
fig1 = plt.figure()
fig1.show()
runrun=0
line = np.array([])
x=[]
y=[]
z=[]
loose_time = 0
is_pass = False
# =================
frame_lose = 0
text = None
cam_hp=[]
success, frame = videoCapture.read()
#--------vediowriter setting--------------
sz = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
vout = cv2.VideoWriter()
vout.open(data_path + '/media'+str(only_one)+ '.mp4',fourcc, 30, sz, True)
passframe = 0

while success :
    if frame.any() ==None :
        break
    frame.flags.writeable = False
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if only_one == 0:
        results = hands.process(frame)
        if results.multi_hand_landmarks :
            for hand_landmarks in results.multi_hand_landmarks:
                x1, y1 = larry_draw(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        else:
            frame_lose +=1
            text = "the frame lose_times: " + str(frame_lose)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # frame = np.flip(frame)
        # if text != None:
        cv2.circle(frame, (0, 0), 30, (0, 255, 255), 3)

        cv2.imshow("frame",frame)

        if (passframe%3)==1:
            pass
        else:
            cam_hp = cam_hp + [x1, y1]
            vout.write(frame)
        passframe += 1


    success, frame = videoCapture.read()
    cv2.waitKey(30) #延迟

if only_one == 0:
    np.save(data_path + "/cam_hp.npy", cam_hp)
    vout.release()
###==============================================================
###==============================================================
###==============================================================
###==============================================================
videoCapture1 =  cv2.VideoCapture(data_path+"vedio2.mp4")
fps1 = videoCapture1.get(cv2.CAP_PROP_FPS)
size1 = (int(videoCapture1.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(videoCapture1.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fNUMS1 = videoCapture1.get(cv2.CAP_PROP_FRAME_COUNT)

red = [0, 0, 255]
red_color = (0, 0, 255) # BGR
cc = 0
fig1 = plt.figure()
fig1.show()
runrun=0
line = np.array([])
x=[]
y=[]
z=[]
loose_time = 0
is_pass = False

# =================
frame1_lose = 0
text1 = None
cam1_hp = []
success1, frame1 = videoCapture1.read()
# =================
only_one = 1
# =================
#--------vediowriter setting--------------
sz = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
vout = cv2.VideoWriter()
vout.open(data_path + '/media'+str(only_one)+ '.mp4',fourcc, 30, sz, True)
passframe = 0
print("Fuck")
while success1 :
    if frame1.any() == None  :
        print("Fuck")
        break
    frame1.flags.writeable = False
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    frame1 = cv2.flip(frame1, 1)
    frame1 = cv2.flip(frame1, 0)

    if only_one == 1:
        results1 = hands.process(frame1)
        if results1.multi_hand_landmarks :
            for hand_landmarks in results1.multi_hand_landmarks:
                x2, y2 = larry_draw(frame1, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(
                    frame1, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        else:
            frame1_lose +=1
            text1 = "the frame lose_times: " + str(frame1_lose)
        # if text1 != None:
        #     cv2.putText(frame1, text1, (10, 430), cv2.FONT_HERSHEY_COMPLEX_SMALL,
        #                 1, (0, 255, 255), 1, cv2.LINE_AA)
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)

        cv2.circle(frame1, (0, 0), 30, (0, 255, 255), 3)

        cv2.imshow("frame1",frame1)

        if (passframe % 3) == 1:
            pass
        else:
            cam1_hp = cam1_hp + [x2, y2]
            vout.write(frame1)
        passframe+=1

    success1, frame1 = videoCapture1.read()
    cv2.waitKey(30) #延迟
if only_one == 1:
    np.save(data_path + "/cam_hp1.npy",cam1_hp)
    vout.release()
