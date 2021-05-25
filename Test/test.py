# import cv2
# import time
# capture = cv2.VideoCapture(0)
# capture.set(cv2.CAP_PROP_FPS, 20)
# print(capture.get(cv2.CAP_PROP_FPS))
# cc = 0
# start = time.time()
# while capture.isOpened():
#
#     ret, frame = capture.read()
#     if ret:
#         cc +=1
#     end = time.time() - start
#     if end != 0:
#         fps = cc/end
#         print(fps)

import cv2
import time

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# cam.set(cv2.CAP_PROP_FPS, 30)
#
# print(cam.get(cv2.CAP_PROP_FPS))

cam.set(cv2.CAP_PROP_FPS, 20)

print(cam.get(cv2.CAP_PROP_FPS))

# cam.set(cv2.CAP_PROP_FPS, 24)
#
# print(cam.get(cv2.CAP_PROP_FPS))
count = 0

start = time.time()
while True:
    ret, img = cam.read()
    count += 1
    vis = img.copy()
    cv2.imshow('getCamera', vis)

    end = time.time() - start
    if count%100 == 0 :
        fps = count/end
        print(fps)
        if fps > 19.9:
            break
    if 0xff & cv2.waitKey(5) == 27:
        stop = time.time()
        break
end_all = time.time()
print("cost time --> {}".format(end_all-start) )
cv2.destroyAllWindows()