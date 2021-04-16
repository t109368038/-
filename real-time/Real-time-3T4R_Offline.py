from offline_process_3t4r import run_proecss
from tkinter import filedialog
import tkinter as tk
import pyqtgraph as pg
import pyqtgraph.ptime as ptime
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication
import pyqtgraph.opengl as gl
import numpy as np
import threading
import sys
import read_binfile
import time
# -----------------------------------------------
from app_layout_2t4r_offline import Ui_MainWindow
from R3t4r_to_point_cloud_for_realtime import plot_pd

class Realtime_sys():
    def __init__(self):
        self.pd_save_status = 0
        self.pd_save = []
        self.rdi = []
        self.rai = []
        self.btn_status = False
        self.run_state  = False
        self.sure_next = True
        self.frame_count = 0

    def start(self):
        self.run_state = True
        self.frame_count = 0
        self.stop_btn.setText("stop")
        self.update_figure()
    def stop(self):
        if self.run_state:
            self.stop_btn.setText("Continues")
        else:
            self.stop_btn.setText("stop")
        self.run_state = not(self.run_state)
        self.sure_next = True
        self.update_figure()

    def RDI_update(self):
        rd = self.RDI
        img_rdi.setImage(np.rot90(rd, -1))

    def RAI_update(self):
        global count, view_rai, p13d
        a = self.RAI
        img_rai.setImage(np.fliplr((a)).T)

    def PD_update(self):
        global count, view_rai, p13d,nice,ax
        # --------------- plot 3d ---------------
        pos = self.PD
        pos = np.transpose(pos,[1,0])
        p13d.setData(pos=pos[:,:3],color= [1,0.35,0.02,1],pxMode= True)

    def update_figure(self):
        global count,view_rai,p13d
        if self.run_state:
            static_removal =self.enable_staic_clutter_removal.isChecked()
            self.RDI ,self.RAI,self.PD = run_proecss(self.rawData[self.frame_count],static_removal)
            self.RDI_update()
            self.RAI_update()
            self.PD_update()
            time.sleep(0.05)
            if self.sure_next:
                self.frame_count +=1
                QtCore.QTimer.singleShot(1, self.update_figure)
                QApplication.processEvents()

        else :
            pass
        print(self.frame_count)

    def pre_frame(self):
        if self.frame_count >0:
            self.frame_count -=1
            self.sure_next = False
            self.run_state=True
            self.update_figure()

    def next_frame(self):
        if self.frame_count<=62:
            self.frame_count -= 1
            self.sure_next = False
            self.run_state = True
            self.update_figure()
    def SelectFolder(self):
        root = tk.Tk()
        root.withdraw()
        self.file_path = filedialog.askopenfilename(parent=root, initialdir='D:\\Matt_yen_data\\NAS\\data\\bin file_processed\\new data(low powered)\\3t4r')
        self.browse_text.setText(self.file_path)

        return self.file_path


    def enable_btns(self,state):
        self.pre_btn.setEnabled(state)
        self.next_btn.setEnabled(state)
        self.start_btn.setEnabled(state)
        self.stop_btn.setEnabled(state)

    def load_file(self):
        self.rawData =read_binfile.read_bin_file(self.file_path,[64,64,32,3,4],mode=0,header=False,packet_num=4322)
        self.rawData = np.transpose(self.rawData,[0,1,3,2])
        print(np.shape(self.rawData))
        self.enable_btns(True)

    def plot(self):
        global img_rdi, img_rai, updateTime, view_text, count, angCurve, ang_cuv, img_cam, savefilename,view_rai,p13d,nice
        # ---------------------------------------------------
        self.app = QtWidgets.QApplication(sys.argv)
        MainWindow = QtWidgets.QMainWindow()
        MainWindow.show()
        ui = Ui_MainWindow()
        ui.setupUi(MainWindow)

        self.browse_btn = ui.browse_btn
        self.browse_text = ui.textEdit
        self.load_btn = ui.load_btn
        self.start_btn = ui.start_btn
        self.stop_btn  =ui.stop_btn
        self.next_btn  =ui.next_btn
        self.pre_btn = ui.pre_btn
        self.enable_staic_clutter_removal = ui.checkBox1
        # #----------------- btn clicked connet -----------------
        self.browse_btn.clicked.connect(self.SelectFolder)
        self.load_btn.clicked.connect(self.load_file)
        self.start_btn.clicked.connect(self.start)
        self.next_btn.clicked.connect(self.next_frame)
        self.pre_btn.clicked.connect(self.pre_frame)
        self.stop_btn.clicked.connect(self.stop)

        self.enable_btns(False)
        # # -----------------------------------------------------
        view_rdi = ui.graphicsView.addViewBox()
        view_rai = ui.graphicsView_2.addViewBox()
        view_PD = ui.graphicsView_3
        # ---------------------------------------------------
        # lock the aspect ratio so pixels are always square
        view_rdi.setAspectLocked(True)
        view_rai.setAspectLocked(True)
        img_rdi = pg.ImageItem(border='w')
        img_rai = pg.ImageItem(border='w')
        img_cam = pg.ImageItem(border='w')
        #-----------------
        xgrid = gl.GLGridItem()
        ygrid = gl.GLGridItem()
        zgrid = gl.GLGridItem()
        view_PD.addItem(xgrid)
        view_PD.addItem(ygrid)
        view_PD.addItem(zgrid)
        xgrid.translate(0,10,-10)
        ygrid.translate(0, 0, 0)
        zgrid.translate(0, 10, -10)
        xgrid.rotate(90, 0, 1, 0)
        ygrid.rotate(90, 1, 0, 0)
        pos = np.random.randint(-10, 10, size=(1000, 3))
        pos[:, 2] = np.abs(pos[:, 2])
        p13d = gl.GLScatterPlotItem(pos = pos,color=[50, 50, 50, 255])
        origin = gl.GLScatterPlotItem(pos = np.zeros([1,3]),color=[255, 0, 0, 255])
        coord = gl.GLAxisItem(glOptions="opaque")
        coord.setSize(10, 10, 10)
        view_PD.addItem(p13d)
        view_PD.addItem(coord)
        view_PD.addItem(origin)

        # ang_cuv = pg.PlotDataItem(tmp_data, pen='r')
        # Colormap
        position = np.arange(64)
        position = position / 64
        position[0] = 0

        position = np.flip(position)
        colors = [[62, 38, 168, 255], [63, 42, 180, 255], [65, 46, 191, 255], [67, 50, 202, 255], [69, 55, 213, 255],
                  [70, 60, 222, 255], [71, 65, 229, 255], [70, 71, 233, 255], [70, 77, 236, 255], [69, 82, 240, 255],
                  [68, 88, 243, 255],
                  [68, 94, 247, 255], [67, 99, 250, 255], [66, 105, 254, 255], [62, 111, 254, 255], [56, 117, 254, 255],
                  [50, 123, 252, 255],
                  [47, 129, 250, 255], [46, 135, 246, 255], [45, 140, 243, 255], [43, 146, 238, 255], [39, 150, 235, 255],
                  [37, 155, 232, 255],
                  [35, 160, 229, 255], [31, 164, 225, 255], [28, 129, 222, 255], [24, 173, 219, 255], [17, 177, 214, 255],
                  [7, 181, 208, 255],
                  [1, 184, 202, 255], [2, 186, 195, 255], [11, 189, 188, 255], [24, 191, 182, 255], [36, 193, 174, 255],
                  [44, 195, 167, 255],
                  [49, 198, 159, 255], [55, 200, 151, 255], [63, 202, 142, 255], [74, 203, 132, 255], [88, 202, 121, 255],
                  [102, 202, 111, 255],
                  [116, 201, 100, 255], [130, 200, 89, 255], [144, 200, 78, 255], [157, 199, 68, 255], [171, 199, 57, 255],
                  [185, 196, 49, 255],
                  [197, 194, 42, 255], [209, 191, 39, 255], [220, 189, 41, 255], [230, 187, 45, 255], [239, 186, 53, 255],
                  [248, 186, 61, 255],
                  [254, 189, 60, 255], [252, 196, 57, 255], [251, 202, 53, 255], [249, 208, 50, 255], [248, 214, 46, 255],
                  [246, 220, 43, 255],
                  [245, 227, 39, 255], [246, 233, 35, 255], [246, 239, 31, 255], [247, 245, 27, 255], [249, 251, 20, 255]]
        colors = np.flip(colors, axis=0)
        color_map = pg.ColorMap(position, colors)
        lookup_table = color_map.getLookupTable(0.0, 1.0, 256)
        img_rdi.setLookupTable(lookup_table)
        img_rai.setLookupTable(lookup_table)
        view_rdi.addItem(img_rdi)
        view_rai.addItem(img_rai)
        view_rdi.setRange(QtCore.QRectF(-5, 0, 140, 80))
        view_rai.setRange(QtCore.QRectF(10, 0, 160, 80))
        updateTime = ptime.time()
        self.app.instance().exec_()
        # print('=======================================')


if __name__ == '__main__':
    print('======Real Time Data Capture Tool======')
    count = 0
    realtime = Realtime_sys()
    lock = threading.Lock()

    plotIMAGE = threading.Thread(target=realtime.plot())
    plotIMAGE.start()

    print("Program Close")
    sys.exit()
