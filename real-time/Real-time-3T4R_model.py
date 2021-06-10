
from tkinter import filedialog
import socket
import tkinter as tk
from queue import Queue
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
from camera_offine import CamCapture
# -----------------------------------------------
from real_time_pd_model import UdpListener, DataProcessor
from radar_config import SerialConfig
from ultis import  send_cmd,get_color,ConnectDca
from Test.Webcam_Save_Vedio_Qthread import RTSPVideoWriterObject
# -----------------------------------------------
from pd_layout_model import Ui_MainWindow
from tensorflow import keras

set_radar = SerialConfig(name='ConnectRadar', CLIPort='COM13', BaudRate=115200)
config = '../radar_config/xwr68xx_profile_2021_03_23T08_12_36_405.cfg'
class ModelTest(QtCore.QThread):
    arr = QtCore.pyqtSignal(np.ndarray)
    def __init__(self, data):
        super(ModelTest, self).__init__()
        self.raw = data
        self.model = keras.models.load_model('D:/Andy-is-very-good/training_data_preprocess/2021-05-31_22-05-47.686357.h5')
        print("load_model_successful")

    def run(self):
        out = self.model.predict(self.raw)
        self.arr.emit(out)

class Realtime_sys():

    def __init__(self):
        adc_sample = 64
        chirp = 16
        tx_num = 3
        rx_num = 4
        self.radar_config = [adc_sample, chirp, tx_num, rx_num]
        frame_length = adc_sample * chirp * tx_num * rx_num * 2
        # Host setting
        address = ('192.168.33.30', 4098)
        buff_size = 2097152
        self.save_frame_len = 1000
        # call class
        self.bindata = Queue()
        self.rawdata = Queue()
        self.collector = UdpListener('Listener', frame_length, address, buff_size,self.bindata,self.rawdata)
        self.camera = False
        self.processor = DataProcessor('Processor', self.radar_config, self.bindata, "0105", status=0 )
        self.processor.data_signal.connect(self.Qthreadupdate_fig)
        self.pd_save_status = 0
        self.pd_save = []
        self.rdi = []
        self.rai = []
        self.raw = []
        self.frame_count = 0
        self.path = 'C:/Users/user/Desktop/thmouse_training_data/'
        #----- for test Usage ----
        self.sure_select = False
        self.rai_mode =  0
        # self.tf_mode = ModelTest(self.raw)
        # self.tf_mode.arr.connect(self.updata_model_out)
        self.model = keras.models.load_model('D:/Andy-is-very-good/training_data_preprocess/2021-05-31_22-05-47.686357.h5')
        self.pd_3 = []
        self.pdQ = Queue()

    def test(self):
        data = self.pdQ.get()
        out_radar_p = self.pd2voxel(data)
        out_radar_p = np.reshape(out_radar_p,[1,3,1,25,25,25])
        out = self.model.predict(out_radar_p)
        print(np.shape(out))
        # self.xz_plane.setData(out[0,:1,:1])

    def pd2voxel(self,data):
        out_radar_p = []

        for i in range(len(data)):
            tmp_frame = data[i]
            arr = np.ndarray([1, 25, 25, 25])
            arr[:] = False

            point_count = 0
            tx = 0
            ty = 0
            tz = 0
            for c in range(tmp_frame.shape[0]):
                x = round(tmp_frame[c][0] / 0.015 + 12.5)
                y = round(tmp_frame[c][1] / 0.015 + 12.5 - 10)
                z = round(tmp_frame[c][2] / 0.015 + 12.5)

                empty = False

                if x < 25 and y < 25 and z < 25 and x > 0 and y > 0 and z > 0:
                    # print("x:{} y:{} z:{}".format(x,y,z))
                    arr[0][x][y][z] = True
                    point_count += 1
                    tx += tmp_frame[c][0]
                    ty += tmp_frame[c][1]
                    tz += tmp_frame[c][2]
            out_radar_p.append(arr)
        return out_radar_p

    def Qthreadupdate_fig(self,rdi,rai,pd):
        self.processor.Sure_staic_RM = self.static_rm.isChecked()
        self.pd_3.append(pd.T)
        if (self.frame_count%3) == 2  :
            self.pdQ.put(self.pd_3)
            self.test()
            self.pd_3 = []
        self.frame_count += 1
        # self.raw.append(self.rawdata.get())
        # if not rai.empty():
        self.img_rdi.setImage(np.rot90(rdi, 1))
        # if not rai.empty():
        self.img_rai.setImage(np.fliplr(np.flip(rai, axis=0)).T)
        # if not pd.empty():
        pos = np.transpose(pd, [1, 0])
        # print(pos)
        p13d.setData(pos=pos[:, :3], color=[1, 0.35, 0.02, 1], pxMode=True)


    def SelectFolder(self):
        root = tk.Tk()
        root.withdraw()
        # self.file_path = filedialog.askopenfilename(parent=root, initialdir='D:\\Matt_yen_data\\NAS\\data\\bin file_processed\\new data(low powered)\\3t4r')
        # self.file_path = filedialog.askopenfilename(parent=root, initialdir='D:/kaiku_report/2021-0418for_posheng/')
        if self.sure_select == True:
            self.file_path = filedialog.askopenfilename(parent=root, initialdir=self.path)
            self.browse_text.setText(self.file_path)
        else:
            self.file_path = self.path + 'raw.npy'
            self.browse_text.setText(self.file_path)

        return self.file_path

    def slot(self, object):
        print("Key was pressed, id is:", self.radio_group.id(object))
        '''
        raimode /0/1/2:
                0 -> FFT-RAI
                1 -> beamformaing RAI 
                3 -> static clutter removal
        '''
        self.rai_mode = self.radio_group.id(object)

        if self.rai_mode ==1:
            self.view_rai.setRange(QtCore.QRectF(10, 0, 170, 80))
        else:
            self.view_rai.setRange(QtCore.QRectF(-5, 0, 100, 60))


    def ConnectDca1000(self):
        global sockConfig, FPGA_address_cfg
        print('Connect to DCA1000')
        print('=======================================')
        config_address = ('192.168.33.30', 4096)
        FPGA_address_cfg = ('192.168.33.180', 4096)
        cmd_order = ['9', 'E', '3', 'B', '5', '6']
        sockConfig = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sockConfig.bind(config_address)
        for k in range(5):
            # Send the command
            sockConfig.sendto(send_cmd(cmd_order[k]), FPGA_address_cfg)
            time.sleep(0.1)
            # Request data back on the config port
            msg, server = sockConfig.recvfrom(2048)
            # print('receive command:', msg.hex())
        sockConfig.close()

    def openradar(self):
        set_radar.StopRadar()
        set_radar.SendConfig(config)
        self.collector.start()
        self.processor.start()
        print('=============openradar=================')

    def exit(self):
        if self.camera == True:
            self.cam1_thread.quit()
            self.cam2_thread.quit()
        self.app.instance().exit()


    def plot(self):
        global img_rdi, img_rai, updateTime, view_text, count, angCurve, ang_cuv, savefilename,view_rai,p13d,nice
        # ---------------------------------------------------
        self.app = QtWidgets.QApplication(sys.argv)
        MainWindow = QtWidgets.QMainWindow()
        MainWindow.show()
        ui = Ui_MainWindow()
        ui.setupUi(MainWindow)
        self.radio_group =  ui.radio_btn_group
        self.static_rm = ui.sure_static

        #----------------- realtime btn clicked connet -----------------
        self.start_dca_rtbtn = ui.dca1000_rtbtn
        self.send_cfg_rtbtn = ui.sendcfg_rtbtn
        self.exit_rtbtn = ui.exit_rtbtn
        #----------------- btn clicked connet -----------------
        self.start_dca_rtbtn.clicked.connect(self.ConnectDca1000)
        self.send_cfg_rtbtn.clicked.connect(self.openradar)
        self.exit_rtbtn.clicked.connect(self.exit)
        self.radio_group.buttonClicked.connect(self.slot)
        # -------------------- view setting ----------------------
        self.view_rdi = ui.graphicsView.addViewBox()
        self.view_rai = ui.graphicsView_2.addViewBox()
        view_PD = ui.graphicsView_3
        self.view_xy = ui.graphicsView_xy
        self.view_xz = ui.graphicsView_xz
        self.view_xyz = ui.graphicsView_xyz
        # lock the aspect ratio so pixels are always square
        self.view_rdi.setAspectLocked(True)
        self.view_rai.setAspectLocked(True)
        self.img_rdi = pg.ImageItem(border='w')
        self.img_rai = pg.ImageItem(border='w')
        self.xy_plane = pg.ScatterPlotItem(pen=pg.mkPen(width=5, color='r'), symbol='o', size=1)
        self.xz_plane = pg.ScatterPlotItem(pen=pg.mkPen(width=5, color='r'), symbol='o', size=1)
        self.view_xy.addItem(self.xy_plane)
        self.view_xz.addItem(self.xz_plane)
        #-------grid setting----
        xgrid = gl.GLGridItem()
        ygrid = gl.GLGridItem()
        zgrid = gl.GLGridItem()
        xgrid.translate(0,10,-10)
        ygrid.translate(0, 0, 0)
        zgrid.translate(0, 10, -10)
        xgrid.rotate(90, 0, 1, 0)
        ygrid.rotate(90, 1, 0, 0)
        p13d = gl.GLScatterPlotItem(pos = np.zeros([1,3]) ,color=[50, 50, 50, 255])
        origin = gl.GLScatterPlotItem(pos = np.zeros([1,3]),color=[255, 0, 0, 255])
        coord = gl.GLAxisItem(glOptions="opaque")
        coord.setSize(10, 10, 10)

        view_PD.addItem(xgrid)
        view_PD.addItem(ygrid)
        view_PD.addItem(zgrid)
        view_PD.addItem(p13d)
        view_PD.addItem(coord)
        view_PD.addItem(origin)
        view_PD.orbit(45,6)
        view_PD.pan(1,1,1,relative=1)
        self.lineup(view_PD)
        self.view_xyz.addItem(xgrid)
        self.view_xyz.addItem(ygrid)
        self.view_xyz.addItem(zgrid)
        self.view_xyz.addItem(coord)
        self.view_xyz.addItem(origin)
        self.view_xyz.orbit(45, 6)
        self.view_xyz.pan(1, 1, 1, relative=1)

        # ang_cuv = pg.PlotDataItem(tmp_data, pen='r')
        # Colormap
        position = np.arange(64)
        position = position / 64
        position[0] = 0
        position = np.flip(position)
        colors = get_color()
        colors = np.flip(colors, axis=0)
        color_map = pg.ColorMap(position, colors)
        lookup_table = color_map.getLookupTable(0.0, 1.0, 256)
        self.img_rdi.setLookupTable(lookup_table)
        self.img_rai.setLookupTable(lookup_table)
        self.view_rdi.addItem(self.img_rdi)
        self.view_rai.addItem(self.img_rai)
        self.view_rdi.setRange(QtCore.QRectF(0, 0, 30, 70))
        self.view_rai.setRange(QtCore.QRectF(10, 0, 160, 80))
        updateTime = ptime.time()
        self.app.instance().exec_()

    def lineup(self,view_PD):
        ###---------------------------------------------
        self.hand = gl.GLScatterPlotItem(pos=np.array([[0, 0, 0]]), color=[0, 255, 0, 255], pxMode=True)
        view_PD.addItem(self.hand)
        self.indexfinger = gl.GLScatterPlotItem(pos=np.array([[0, 0, 0]]), color=[255, 0, 0, 255], pxMode=True)
        view_PD.addItem(self.indexfinger)
        self.thumb = gl.GLScatterPlotItem(pos=np.array([[0, 0, 0]]), color=[255, 0, 0, 255], pxMode=True)
        view_PD.addItem(self.thumb)
        origin = gl.GLScatterPlotItem(pos=np.array(
            [[0, 0.075, 0], [0, 0.075 * 2, 0], [0, 0.075 * 3, 0], [0, 0.075 * 4, 0], [0, 0.075 * 5, 0],
             [0, 0.075 * 6, 0]]), color=[255, 255, 255, 255])
        origin1 = gl.GLScatterPlotItem(pos=np.array(
            [[0.075 * -3, 0.3, 0], [0.075 * -2, 0.3, 0], [0.075 * -1, 0.3, 0], [0.075 * 1, 0.3, 0],
             [0.075 * 2, 0.3, 0], [0.075 * 3, 0.3, 0]]), color=[255, 255, 255, 255])
        origin2 = gl.GLScatterPlotItem(pos=np.array(
            [[0, 0.3, 0.075 * -3], [0, 0.3, 0.075 * -2], [0, 0.3, 0.075 * -1], [0, 0.3, 0.075 * 1],
             [0, 0.3, 0.075 * 2], [0, 0.3, 0.075 * 3]]), color=[255, 255, 255, 255])
        view_PD.addItem(origin)
        view_PD.addItem(origin1)
        view_PD.addItem(origin2)
        origin_P = gl.GLScatterPlotItem(pos=np.array(
            [[0, 0, 0]]), color=[255, 0, 0, 255])
        view_PD.addItem(origin_P)
        self.hand_line = gl.GLLinePlotItem(pos=np.array([[[0, 0, 0], [0, 0.075 * 10, 0]]]),
                                           color=[128, 255, 128, 255], antialias=False)
        self.hand_liney = gl.GLLinePlotItem(pos=np.array([[[0, 0, 0], [0, 0.075 * 10, 0]]]),
                                            color=[128, 255, 128, 255], antialias=False)
        self.hand_linex_2 = gl.GLLinePlotItem(pos=np.array([[[-0.075 * 8, 0.075 * 4, 0.075 * 2], [0.075 * 8, 0.075 * 4, 0.075 * 2]]]),
                                              color=[128, 255, 128, 255], antialias=False)
        self.hand_linex_1 = gl.GLLinePlotItem(pos=np.array([[[-0.075 * 8, 0.075 * 4, 0.075*1], [0.075 * 8, 0.075 * 4, 0.075*1]]]),
                                            color=[128, 255, 128, 255], antialias=False)
        self.hand_linex_d1 = gl.GLLinePlotItem(pos=np.array([[[-0.075 * 8, 0.075 * 4, 0.075 * -1], [0.075 * 8, 0.075 * 4, 0.075 * -1]]]),
                                            color=[128, 255, 128, 255], antialias=False)
        self.hand_linex_d2 = gl.GLLinePlotItem(pos=np.array([[[-0.075 * 8, 0.075 * 4, 0.075 * -2], [0.075 * 8, 0.075 * 4, 0.075 * -2]]]),
                                            color=[128, 255, 128, 255], antialias=False)

        self.hand_linex = gl.GLLinePlotItem(pos=np.array([[[-0.075 * 8, 0.075 * 4, 0], [0.075 * 8, 0.075 * 4, 0]]]),
                                            color=[128, 255, 128, 255], antialias=False)

        self.hand_linez = gl.GLLinePlotItem(pos=np.array([[[0, 0.075 * 4, -0.075 * 8], [0, 0.075 * 4, 0.075 * 8]]]),
                                            color=[0.5,0.5,0.9,1], antialias=False)
        view_PD.addItem(self.hand_line)
        view_PD.addItem(self.hand_liney)
        view_PD.addItem(self.hand_linez)
        view_PD.addItem(self.hand_linex)



if __name__ == '__main__':
    print('======Real Time Data Capture Tool======')
    count = 0
    realtime = Realtime_sys()
    lock = threading.Lock()

    plotIMAGE = threading.Thread(target=realtime.plot())
    plotIMAGE.start()
    print("Program Close")

    set_radar.StopRadar()
    sys.exit()