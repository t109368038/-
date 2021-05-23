import time
import threading
import cv2
class AddDaemon(object):
    def __init__(self):
        self.stuff = 'th1'

    def add(self):
        while True:
            print(self.stuff)
            time.sleep(1)


class RemoveDaemon(object):
    def __init__(self):
        self.stuff = 'th2'

    def rem(self):
        while True:
            print(self.stuff)
            time.sleep(1)
class RemoveDaemonx(object):
    def __init__(self):
        self.stuff = 'th3'
        self.capture = cv2.VideoCapture(0)
    def rem(self):
        while True:
            if self.capture.isOpened():
                ret,frame = self.capture.read()
                print(frame)

def main():
    a = AddDaemon()
    r = RemoveDaemon()
    c = RemoveDaemonx()
    t1 = threading.Thread(target=r.rem)
    t2 = threading.Thread(target=a.add)
    t3 = threading.Thread(target=c.rem)
    t1.setDaemon(True)
    t2.setDaemon(True)
    t3.setDaemon(True)
    t1.start()
    t2.start()
    t3.start()
    time.sleep(1000)

if __name__ == '__main__':
    main()