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
        self.w.setGeometry(0, 110, 700, 600)
        self.w.show()

        # create the background grids
        gx = gl.GLGridItem()
        gx.translate(0, 10, -10)
        gx.rotate(90, 0, 1, 0)
        self.w.addItem(gx)

        gy = gl.GLGridItem()
        gy.translate(0, 0, 0)
        gy.rotate(90, 1, 0, 0)
        self.w.addItem(gy)

        gz = gl.GLGridItem()
        gz.translate(0, 10, -10)
        self.w.addItem(gz)

        self.origin = gl.GLScatterPlotItem(pos=np.zeros([1, 3]), color=[255, 0, 0, 255], pxMode=True)
        self.w.addItem(self.origin)

        self.traces = gl.GLScatterPlotItem(pos=np.array([[0, 0, 0]]), color=[255, 255, 255, 255], pxMode=True)
        self.w.addItem(self.traces)

    def start(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()

    def set_plotdata(self):
        if not self.RadarData.empty():
            data = self.RadarData.get()
            data = data[:, :3]
            data = data.reshape([-1, 3])
            self.traces.setData(pos=data, color=[255, 255, 255, 255], pxMode=True)

    def update(self):
            self.set_plotdata()

    def animation(self):
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(50)
        self.start()


# Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    path = 'E:/NTUT-master/KaiKu Report/0426/'
    radar_file = 'fingermovepd.npy'
    radar_pd = np.load(path + radar_file, allow_pickle=True)
    radar_pos = queue.Queue()
    for j, pos in enumerate(radar_pd):
        radar_pos.put(pos)

    v = Visualizer(radar_pos)
    v.animation()
