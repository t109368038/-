import queue
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import pyqtgraph as pg
import numpy as np
import sys
import time


class Visualizer(object):
    def __init__(self, RadarData=None, ImgData=None):
        self.RadarData = RadarData
        self.ImgData = ImgData
        self.traces = dict()
        self.app = QtGui.QApplication(sys.argv)
        self.w = gl.GLViewWidget()
        self.w.opts['distance'] = 10
        self.w.setWindowTitle('Combine Radar and Image')
        self.w.setGeometry(0, 110, 1920, 1080)
        self.w.show()

        # # create the background grids
        # gx = gl.GLGridItem()
        # gx.translate(0, 10, -10)
        # gx.rotate(90, 0, 1, 0)
        # self.w.addItem(gx)

        gy = gl.GLGridItem()
        gy.translate(0, 0, 0)
        gy.rotate(90, 1, 0, 0)
        self.w.addItem(gy)

        # gz = gl.GLGridItem()
        # gz.translate(0, 10, -10)
        # self.w.addItem(gz)

        self.origin = gl.GLScatterPlotItem(pos=np.zeros([1, 3]), color=[255, 0, 0, 255], pxMode=True)
        self.w.addItem(self.origin)

        self.xaxispoint = gl.GLScatterPlotItem(pos=np.array([[0.075 * -4, 0.3, 0], [0.075 * -3, 0.3, 0],
                                                             [0.075 * -2, 0.3, 0], [0.075 * -1, 0.3, 0],
                                                             [0.075 * 1, 0.3, 0], [0.075 * 2, 0.3, 0],
                                                             [0.075 * 3, 0.3, 0], [0.075 * 4, 0.3, 0]]),
                                               color=[255, 255, 255, 128], pxMode=True)
        self.w.addItem(self.xaxispoint)
        self.xaxisline = gl.GLLinePlotItem(pos=np.array([[0.075 * -4, 0.3, 0], [0.075 * 4, 0.3, 0]]),
                                           color=[0.99, 0.65, 0.5, 128], antialias=False)
        self.w.addItem(self.xaxisline)

        self.yaxispoint = gl.GLScatterPlotItem(pos=np.array([[0, 0.075 * 1, 0], [0, 0.075 * 2, 0],
                                                             [0, 0.075 * 3, 0], [0, 0.075 * 4, 0],
                                                             [0, 0.075 * 4, 0], [0, 0.075 * 5, 0],
                                                             [0, 0.075 * 6, 0], [0, 0.075 * 7, 0],
                                                             [0, 0.075 * 8, 0]]),
                                               color=[255, 255, 255, 128], pxMode=True)
        self.w.addItem(self.yaxispoint)
        self.yaxisline = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, 0.075 * 8, 0]]),
                                           color=[0.49, 0.25, 0.77, 128], antialias=False)
        self.w.addItem(self.yaxisline)

        self.zaxispoint = gl.GLScatterPlotItem(pos=np.array([[0, 0.3, 0.075 * -4], [0, 0.3, 0.075 * -3],
                                                             [0, 0.3, 0.075 * -2], [0, 0.3, 0.075 * -1],
                                                             [0, 0.3, 0.075 * 1], [0, 0.3, 0.075 * 2],
                                                             [0, 0.3, 0.075 * 3], [0, 0.3, 0.075 * 4]]),
                                               color=[255, 255, 255, 128], pxMode=True)
        self.w.addItem(self.zaxispoint)
        self.zaxisline = gl.GLLinePlotItem(pos=np.array([[0, 0.3, 0.075 * -4], [0, 0.3, 0.075 * 4]]),
                                           color=[0.21, 0.44, 0.66, 0.64], antialias=False)
        self.w.addItem(self.zaxisline)

        self.traces = gl.GLScatterPlotItem(pos=np.array([[0, 0, 0]]), color=[0, 0, 255, 255], pxMode=True)
        self.w.addItem(self.traces)

        self.hand = gl.GLScatterPlotItem(pos=np.array([[0, 0, 0]]), color=[0, 255, 0, 255], pxMode=True)
        self.w.addItem(self.hand)

        self.hand_line = gl.GLLinePlotItem(pos=np.array([[[0, 0, 0], [2.5, 3.2, 1.5]], [[0, 0, 0], [1, 3.5, 4]]]),
                                           color=[128, 255, 128, 255], antialias=False)
        self.w.addItem(self.hand_line)

        self.indexfinger = gl.GLScatterPlotItem(pos=np.array([[0, 0, 0]]), color=[0.35, 0.62, 0.35, 255], pxMode=True)
        self.w.addItem(self.indexfinger)

        self.thumb = gl.GLScatterPlotItem(pos=np.array([[0, 0, 0]]), color=[255, 0, 0, 255], pxMode=True)
        self.w.addItem(self.thumb)

        # self.coord = gl.GLAxisItem(glOptions="opaque")
        # self.coord.setSize(10, 10, 10)
        # self.w.addItem(self.coord)

    def start(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()

    def set_plotdata(self):
        if not self.RadarData.empty():
            data = self.RadarData.get()
            data = data[:, :3]
            data = data.reshape([-1, 3])
            self.traces.setData(pos=data, color=[0, 0, 255, 255], pxMode=True)

            hand = self.ImgData.get()
            # print("hand_shape: {}".format(hand.shape))
            hand = hand.transpose([1, 0])
            self.hand.setData(pos=hand, color=[0, 255, 0, 255], pxMode=True)

            line = np.array(
                [[hand[0, :], hand[1, :]], [hand[1, :], hand[2, :]], [hand[2, :], hand[3, :]], [hand[3, :], hand[4, :]],
                 [hand[0, :], hand[5, :]],
                 [hand[5, :], hand[6, :]], [hand[6, :], hand[7, :]], [hand[7, :], hand[8, :]], [hand[5, :], hand[9, :]],
                 [hand[9, :], hand[10, :]],
                 [hand[10, :], hand[11, :]], [hand[11, :], hand[12, :]], [hand[9, :], hand[13, :]],
                 [hand[13, :], hand[14, :]], [hand[14, :], hand[15, :]],
                 [hand[15, :], hand[16, :]], [hand[13, :], hand[17, :]], [hand[17, :], hand[18, :]],
                 [hand[18, :], hand[19, :]], [hand[19, :], hand[20, :]],
                 [hand[0, :], hand[17, :]]])

            self.hand_line.setData(pos=line, color=[0.5, 0.7, 0.9, 255], antialias=False)

            self.indexfinger.setData(pos=hand[8, :], color=[0.88, 0.22, 0.35, 255], pxMode=True)
            self.thumb.setData(pos=hand[4, :], color=pg.glColor((255, 255, 0)), pxMode=True)

    def update(self):
        self.set_plotdata()

    def animation(self):
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(50)
        self.start()


def GetGroundTruth(x1, y1, x2, y2):
    scale_x1 = 56 / 640 * 0.015  # cm / pixel * (0.015 point/cm)
    scale_y1 = 44 / 480 * 0.015
    scale_x2 = 45 / 640 * 0.015
    scale_y2 = 33 / 480 * 0.015

    if (y1 != None).all() == True:
        y1 = y1.astype(np.double)
        y1 = np.round((y1 * scale_y1), 3)
        y1 -= ((480 * scale_y1) / 2)

    x2 = np.where(x2 is not None, x2, np.double(320))
    x2 = x2.astype(np.double)
    x2 = (x2 * scale_x2)
    x2 -= ((640 * scale_x2) / 2)
    x2 = np.round(x2, 3)

    y2 = np.where(y2 is not None, y2, np.double(480))
    y2 = y2.astype(np.double)
    y2 = (y2 * scale_y2)
    y2 -= ((480 * scale_y2) / 2)
    y2 = np.round(y2, 3)

    return x2 * -1, y2 + 0.33, y1 * -1


# Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    path = 'E:/NTUT-master/KaiKu Report/0426/write8/'
    # path = 'E:/NTUT-master/KaiKu Report/0426/elv/'
    # path = 'E:/NTUT-master/KaiKu Report/0426/range/'
    radar_file = 'rm_pd.npy'
    hand_file_1 = 'cam_hp.npy'
    hand_file_2 = 'cam_hp1.npy'

    radar_pd = np.load(path + radar_file, allow_pickle=True)
    hand_pd_1 = np.load(path + hand_file_1, allow_pickle=True)
    hand_pd_2 = np.load(path + hand_file_2, allow_pickle=True)

    hand_pd_pos = queue.Queue()

    for h in range(int(hand_pd_1.shape[0] / 2)):
        data = GetGroundTruth(hand_pd_1[h * 2, :], hand_pd_1[h * 2 + 1, :],
                              hand_pd_2[h * 2, :], hand_pd_2[h * 2 + 1, :])
        hand_pd_pos.put(np.array(data))

    radar_pos = queue.Queue()
    for j, pos in enumerate(radar_pd):
        radar_pos.put(pos)

    v = Visualizer(radar_pos, hand_pd_pos)
    v.animation()
