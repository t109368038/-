import cv2
import mediapipe as mp
import numpy as np

image_width = 500
image_high = 600
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# For webcam input:
# hands = mp_hands.Hands(static_image_mode=True,
#     min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=1)
# cap = cv2.VideoCapture(0)
# while cap.isOpened():
#     success, image = cap.read()
#     if not success:
#         print("Ignoring empty camera frame.")
#         # If loading a video, use 'break' instead of 'continue'.
#         continue
#
#     # Flip the image horizontally for a later selfie-view display, and convert
#     # the BGR image to RGB.
#     image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
#     # To improve performance, optionally mark the image as not writeable to
#     # pass by reference.
#     image.flags.writeable = False
#     results = hands.process(image)
#
#     # Draw the hand annotations on the image.
#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(
#                 image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#             # print('hand_landmarks:', hand_landmarks)
#             print(
#                 f'Index finger tip coordinates: (',
#                 f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width},'
#                 f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_high})'
#             )
#             print(
#                 f'Thumb tip coordinates: (',
#                 f'{hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width}, '
#                 f'{hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_high})'
#             )
#             print('\n')
#     cv2.imshow('MediaPipe Hands', image)
#     if cv2.waitKey(5) & 0xFF == 27:
#         break
# hands.close()
# cap.release()

data = np.load('E:/ResearchData/ThuMouseData/handtest_cam2.npy')
with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:
  for idx, file in enumerate(data):
    # Read an image, flip it around y-axis for correct handedness output (see
    # above).
    # image = cv2.flip(cv2.imread(file), 1)
    image = file
    # Convert the BGR image to RGB before processing.
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print handedness and draw hand landmarks on the image.
    print('Handedness:', results.multi_handedness)
    if not results.multi_hand_landmarks:
      continue
    image_height, image_width, _ = image.shape
    annotated_image = image.copy()
    for hand_landmarks in results.multi_hand_landmarks:
      print('hand_landmarks:', hand_landmarks)
      print(
          f'Index finger tip coordinates: (',
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
      )
      mp_drawing.draw_landmarks(
          annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    # cv2.imwrite(
    #     '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break


# For webcam input:
# cap = cv2.VideoCapture(0)
# with mp_hands.Hands(
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5) as hands:
#   while cap.isOpened():
#     # success, image = cap.read()
#     for i, frame in enumerate(data):
#         image = frame
#         # if not success:
#         #   print("Ignoring empty camera frame.")
#         #   # If loading a video, use 'break' instead of 'continue'.
#         #   continue
#
#         # Flip the image horizontally for a later selfie-view display, and convert
#         # the BGR image to RGB.
#         image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
#         # To improve performance, optionally mark the image as not writeable to
#         # pass by reference.
#         image.flags.writeable = False
#         results = hands.process(image)
#
#         # Draw the hand annotations on the image.
#         image.flags.writeable = True
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#         if results.multi_hand_landmarks:
#           for hand_landmarks in results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(
#                 image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#         cv2.imshow('MediaPipe Hands', image)
#         if cv2.waitKey(5) & 0xFF == 27:
#           break
# # cap.release()

