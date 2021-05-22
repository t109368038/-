import cv2
from time import time as timer
import sys
import numpy as np


# 開啟影片檔案
# path = 'C:\\Users\\user\\Desktop\\thmouse_training_data\\vedio1.mp4'
path = 'C:\\Users\\user\\Desktop\\thmouse_training_data\\WIN_20210521_12_22_32_Pro.mp4'
cap = cv2.VideoCapture(path)
framerate = timer()

fps = 30
# fps = cap.get(cv2.CAP_PROP_FPS)
print("fps :{}".format(fps))
# fps /= 1000
fps =  1/fps
elapsed = int()

print(cap)
cc = 0
# 以迴圈從影片檔案讀取影格，並顯示出來
while(1):
    if elapsed == 120:
        break
    start = timer()

    ret, frame = cap.read()
    cc += 1
    cv2.imshow('frame',frame)

    diff = timer() - start
    while diff < fps:
        diff = timer() - start
        y=timer()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elapsed += 1
    # if elapsed % 5 == 0:
    #     sys.stdout.write('\r')
    #     taketime=timer() - 0.0001 * 120 - framerate
    #     sys.stdout.write('{0:3.3f} FPS'.format(elapsed / (taketime)))
    #     sys.stdout.write('   diff: {}'.format(diff))
    #     sys.stdout.flush()
# print(taketime)
sys.stdout.write('\r')
taketime=timer()  - framerate
sys.stdout.write('{0:3.3f} FPS'.format(elapsed / (taketime)))
# sys.stdout.write('   diff: {}'.format(diff))
sys.stdout.flush()
cap.release()
cv2.destroyAllWindows()