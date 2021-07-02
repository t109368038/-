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

def save_mediapipe_point(path, save_path, mp_hands):
    # =================
    hands = mp_hands.Hands(
        # static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    videoCapture  =  cv2.VideoCapture(path+"vedio1.mp4")
    # =================
    only_one = 0
    # =================
    frame_lose = 0
    passframe = 0
    cam_hp=[]
    success, frame = videoCapture.read()

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
            else:
                frame_lose +=1

            cam_hp = cam_hp + [x1, y1]
            passframe += 1


        success, frame = videoCapture.read()
        cv2.waitKey(30) #延迟

    if only_one == 0:
        np.save(save_path + "cam_hp.npy", cam_hp)
        print("   video 1 process Done")
#----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
    hands = mp_hands.Hands(
        # static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        )

    videoCapture1 =  cv2.VideoCapture(path+"vedio2.mp4")
    # =================
    frame1_lose = 0
    cam1_hp = []
    success1, frame1 = videoCapture1.read()
    # =================
    only_one = 1

    while success1 :
        if frame1.any() == None  :
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
            else:
                frame1_lose +=1
                x2 = None
                y2 = None

            cam1_hp = cam1_hp + [x2, y2]
        success1, frame1 = videoCapture1.read()
        cv2.waitKey(30) #延迟

    if only_one == 1:
        np.save(path + "cam_hp1.npy",cam1_hp)
        print("   video 1 process Done")

