import cv2
import sys
import math
import keyboard
import argparse
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from collections import deque
from typing import List, Tuple, Union
from mediapipe_proecss import _normalized_to_pixel_coordinates,larry_draw,plot_hand
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)

cap = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)

with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened() and cap1.isOpened():
    success, image = cap.read()
    success1, image1 = cap1.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue
    if not success1:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue
    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image1 = cv2.cvtColor(cv2.flip(image1, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reflags.ference.
    image.flags.writeable = False
    image1.flags.writeable = False
    image_v = cv2.vconcat([image, image1])
    results = hands.process(image_v)

    # Draw the hand annotations on the image.
    image_v.flags.writeable = True
    image_v = cv2.cvtColor(image_v, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image_v, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    cv2.imshow('MediaPipe Hands', image_v)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()