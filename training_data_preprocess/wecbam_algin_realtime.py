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


def lineup_center(image):
    cv2.line(image, (0, 239), (640, 239), (0, 0, 255), 2)
    cv2.line(image, (0, 240), (640, 240), (0, 0, 255), 2)
    cv2.line(image, (319, 0), (319, 480), (0, 0, 255), 2)
    cv2.line(image, (320, 0), (320, 480), (0, 0, 255), 2)
    return image


tmp_path = 'C:/Users/user/Desktop/thmouse_training_data/'
videoCapture  =  cv2.VideoCapture(0)
videoCapture1 = cv2.VideoCapture(1)

while True :
    success, frame = videoCapture.read()
    success1, frame1 = videoCapture1.read()

    frame = lineup_center(frame)
    frame1 = lineup_center(frame1)
    cv2.imshow("frame", frame)
    cv2.imshow("frame1", frame1)
    cv2.waitKey(1) #延迟
