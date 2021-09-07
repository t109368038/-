import DSP_2t4r
from real_time_process_3t4r import UdpListener, DataProcessor, class_predict
from CameraCapture import CamCapture
from scipy import signal
from radar_config import SerialConfig
from queue import Queue
from tkinter import filedialog
import tkinter as tk
import queue
import serial
import pyqtgraph as pg
import pyqtgraph.ptime as ptime
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import numpy as np
import threading
import time
import sys
import socket
import cv2
import matplotlib.pyplot as plt
import pyqtgraph.opengl as gl
import csv
import os
# -----------------------------------------------
from app_layout_2t4r import Ui_MainWindow
from tensorflow.keras.models import Sequential, load_model
# -----------------------------------------------
# config = '../radar_config/IWR1843_cfg_3t4r_v3.4_1.cfg'
config = '../radar_config/xwr68xx_profile_2021_03_23T08_12_36_405.cfg'
# config = '../radar_config/xwr18xx_profile_2021_03_09T10_45_11_974.cfg'
# config = '../radar_config/IWR1843_3d.cfg'
# config = '../radar_config/xwr18xx_profile_2021_03_05T07_10_37_413.cfg'

set_radar = SerialConfig(name='ConnectRadar', CLIPort='COM10', BaudRate=115200)
# set_radar = SerialConfig(name='ConnectRadar', CLIPort='COM13', BaudRate=115200)


def send_cmd(code):
    # command code list
    CODE_1 = (0x01).to_bytes(2, byteorder='little', signed=False)
    CODE_2 = (0x02).to_bytes(2, byteorder='little', signed=False)
    CODE_3 = (0x03).to_bytes(2, byteorder='little', signed=False)
    CODE_4 = (0x04).to_bytes(2, byteorder='little', signed=False)
    CODE_5 = (0x05).to_bytes(2, byteorder='little', signed=False)
    CODE_6 = (0x06).to_bytes(2, byteorder='little', signed=False)
    CODE_7 = (0x07).to_bytes(2, byteorder='little', signed=False)
    CODE_8 = (0x08).to_bytes(2, byteorder='little', signed=False)
    CODE_9 = (0x09).to_bytes(2, byteorder='little', signed=False)
    CODE_A = (0x0A).to_bytes(2, byteorder='little', signed=False)
    CODE_B = (0x0B).to_bytes(2, byteorder='little', signed=False)
    CODE_C = (0x0C).to_bytes(2, byteorder='little', signed=False)
    CODE_D = (0x0D).to_bytes(2, byteorder='little', signed=False)
    CODE_E = (0x0E).to_bytes(2, byteorder='little', signed=False)

    # packet header & footer
    header = (0xA55A).to_bytes(2, byteorder='little', signed=False)
    footer = (0xEEAA).to_bytes(2, byteorder='little', signed=False)

    # data size
    dataSize_0 = (0x00).to_bytes(2, byteorder='little', signed=False)
    dataSize_6 = (0x06).to_bytes(2, byteorder='little', signed=False)

    # data
    data_FPGA_config = (0x01020102031e).to_bytes(6, byteorder='big', signed=False) # lvds 4
    # data_FPGA_config = (0x01020102011e).to_bytes(6, byteorder='big', signed=False)   # lvds 2
    data_packet_config = (0xc005350c0000).to_bytes(6, byteorder='big', signed=False)

    # connect to DCA1000
    connect_to_FPGA = header + CODE_9 + dataSize_0 + footer
    read_FPGA_version = header + CODE_E + dataSize_0 + footer
    config_FPGA = header + CODE_3 + dataSize_6 + data_FPGA_config + footer
    config_packet = header + CODE_B + dataSize_6 + data_packet_config + footer
    start_record = header + CODE_5 + dataSize_0 + footer
    stop_record = header + CODE_6 + dataSize_0 + footer

    if code == '9':
        re = connect_to_FPGA
    elif code == 'E':
        re = read_FPGA_version
    elif code == '3':
        re = config_FPGA
    elif code == 'B':
        re = config_packet
    elif code == '5':
        re = start_record
    elif code == '6':
        re = stop_record
    else:
        re = 'NULL'
    # print('send command:', re.hex())
    return re


def update_figure():
    global count, previous_status, is_hand_there, frame_buffer, predict_result, true_label, frame_num_1
    true_label = text_label.toPlainText()
    win_param = [8, 8, 3, 3]
    # cfar_rai = CA_CFAR(win_param, threshold=2.5, rd_size=[64, 181])
    if not RDIData.empty():
        rd = RDIData.get()
        if ((np.mean(rd) - previous_status) > 0.9) and ((np.mean(rd) - previous_status) < 20):
            if is_hand_there == 0:
                print('There is a hand')
                is_hand_there = 1
                StartRecord()
                count += 1

        savefilename.setText('Current Power:{}'.format(np.mean(rd)))
        predict_count.setText('Predict Times:{}'.format(count))
        previous_status = np.mean(rd)
        # if len(frame_buffer) != 8:
        #     frame_buffer.append(rd)
        # else:
        #     frame_buffer.append(rd)
        #     frame_buffer = frame_buffer[1:]
        #     # print(np.shape(frame_buffer))



        # img_rdi.setImage(rd[:, :, 0].T, levels=[0, 2.6e4])
        # img_rdi.setImage(np.rot90(np.fft.fftshift(rd, axes=1), 3))
        img_rdi.setImage(rd)
        # img_rdi.setImage(np.abs(RDIData.get()[:, :, 0].T))
        # img_rai.setImage(cfar_rai(np.fliplr(RAIData.get()[0, :, :])).T)
        # ang_cuv.setData(rd[:, :, 0].sum(1))

    # if not RAIData.empty():
        # xx = RAIData.get()

        # pd = RAIData.get()
        # pos = np.transpose(pd, [1, 0])
        # p13d.setData(pos=pos[:, :3], color=[1, 0.35, 0.02, 1], pxMode=True)
        # img_rai.setImage(xx)

        # img_rai.setImage(np.fliplr(np.flip(xx, axis=0)).T)
        # np.save('../data/0105/rai_new' + str(count), xx[36:-1, :])

        # img_rai.setImage(np.fliplr(np.flip(xx, axis=0)).T, levels=[1.2e4, 4e6])
        # angCurve.plot((np.fliplr(np.flip(xx, axis=0)).T)[:, 10:12].sum(1), clear=True)

    # if not CAMData.empty():
    #     yy = CAMData.get()
    #     # np.save('../data/img/' + str(count), yy)
    #     img_cam.setImage(np.rot90(yy, -1))

    # print('record status', collector.record_status)
    # print('save status', collector.status)
    # print(collector.count_frame)
    if collector.count_frame == frame_num_1:
        collector.count_frame = 0
        StopRecord()
        is_hand_there = 0
    QtCore.QTimer.singleShot(1, update_figure)
    now = ptime.time()
    updateTime = now


def openradar():
    # if not CAMData.empty():
    #     set_radar.StopRadar()
    #     set_radar.SendConfig(config)
    #     update_figure()
    set_radar.StopRadar()
    set_radar.SendConfig(config)
    print('=======================================')
    update_figure()


def StartRecord():
    # processor.status = 1
    collector.status = 1
    # cam1.status = 1
    # cam2.status = 1
    print('Start Record Time:', (time.ctime(time.time())))
    print('=======================================')


def StopRecord():
    # set_radar.StopRadar()
    # processor.status = 0
    collector.status = 0
    # cam1.status = 0
    # cam2.status = 0
    print('Stop Record Time:', (time.ctime(time.time())))
    print('=======================================')
    collector.record_status == 0
    model_predict()

    while not rawData.empty():
        tmp = rawData.get()
    while not cam_rawData.empty():
        tmp = cam_rawData.get()
    while not cam_rawData2.empty():
        tmp = cam_rawData2.get()
    while not BinData.empty():
        tmp = BinData.get()


def ConnectDca():
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


def SelectFolder():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.asksaveasfilename(parent=root, initialdir='C:\\Users\\user\\Desktop\\ForAndy')
    return file_path


def SaveData():
    global savefilename, sockConfig, FPGA_address_cfg
    set_radar.StopRadar()
    # sockConfig.sendto(send_cmd('6'), FPGA_address_cfg)
    # sockConfig.close()
    path = SelectFolder()
    # cam1.status = 0
    # cam2.status = 0
    if path:
        savefilename.setText('Save File Path & Name: ' + path)
        # name = savefilename.toPlainText()
        print('=======================================')
        print('File Save Radar:', path + '_rawdata')
        # print('File Same Cam1:', path + '_cam1')
        # print('File Same Cam2:', path + '_cam2')
        print('=======================================')
        data_save = []
        while not rawData.empty():
            tmp = rawData.get()
            data_save.append(tmp)
        np.save(path + '_rawdata', data_save)
        print('Radar File Size', np.shape(data_save)[0])
        print('=======================================')
        cam_save = []
        while not cam_rawData.empty():
            tmp = cam_rawData.get()
            cam_save.append(tmp)
        np.save(path + '_cam1', cam_save)
        print('Camera 1 File Size:', np.shape(cam_save)[0])
        print('=======================================')
        cam_save2 = []
        while not cam_rawData2.empty():
            tmp = cam_rawData2.get()
            cam_save2.append(tmp)
        np.save(path  + '_cam2', cam_save2)
        print('Camera 2 File Size:', np.shape(cam_save2)[0])
        print('=======================================')
        print('Save File Done')
        print('=======================================')

        while not BinData.empty():
            zz = BinData.get()
        print('Clear Bin Queue')
        print('=======================================')

        img_rdi.clear()
        # img_cam.clear()


def plot():
    global img_rdi, img_rai, updateTime, view_text, count, angCurve, ang_cuv, img_cam, savefilename, p13d, gesture_result, text_label, predict_count, text_label
    # ---------------------------------------------------
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    MainWindow.show()
    tmp_data = np.zeros(181)
    # angCurve = pg.plot(tmp_data, pen='r')
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    view_rdi = ui.graphicsView.addViewBox()
    view_rai = ui.graphicsView_2.addViewBox()
    # view_cam = ui.graphicsView_3.addViewBox()
    view_pd = ui.graphicsView_3

    # view_angCurve = ui.graphicsView_3.addViewBox()

    starbtn = ui.pushButton_start
    exitbtn = ui.pushButton_exit
    recordbtn = ui.pushButton_record
    stoprecordbtn = ui.pushButton_stop_record
    savebtn = ui.pushButton_save
    dcabtn = ui.pushButton_DCA
    savefilename = ui.label_3
    gesture_result = ui.label_5
    predict_count = ui.label_4
    text_label = ui.textEdit
    # ---------------------------------------------------
    # lock the aspect ratio so pixels are always square
    view_rdi.setAspectLocked(True)
    view_rai.setAspectLocked(True)
    # view_cam.setAspectLocked(True)
    img_rdi = pg.ImageItem(border='w')
    img_rai = pg.ImageItem(border='w')
    # img_cam = pg.ImageItem(border='w')
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
    # view_cam.addItem(img_cam)
    # view_angCurve.addItem(ang_cuv)

    xgrid = gl.GLGridItem()
    ygrid = gl.GLGridItem()
    zgrid = gl.GLGridItem()
    view_pd.addItem(xgrid)
    view_pd.addItem(ygrid)
    view_pd.addItem(zgrid)
    xgrid.translate(0, 10, -10)
    ygrid.translate(0, 0, 0)
    zgrid.translate(0, 10, -10)
    xgrid.rotate(90, 0, 1, 0)
    ygrid.rotate(90, 1, 0, 0)

    p13d = gl.GLScatterPlotItem(pos=np.zeros([1, 3]), color=[50, 50, 50, 255])
    origin = gl.GLScatterPlotItem(pos=np.zeros([1, 3]), color=[255, 0, 0, 255])
    coord = gl.GLAxisItem(glOptions="opaque")
    coord.setSize(10, 10, 10)
    view_pd.addItem(p13d)
    view_pd.addItem(coord)
    view_pd.addItem(origin)

    # Set initial view bounds
    view_rdi.setRange(QtCore.QRectF(-5, 0, 140, 80))
    view_rai.setRange(QtCore.QRectF(10, 0, 160, 80))
    updateTime = ptime.time()
    starbtn.clicked.connect(openradar)
    recordbtn.clicked.connect(StartRecord)
    stoprecordbtn.clicked.connect(StopRecord)
    savebtn.clicked.connect(SaveData)
    dcabtn.clicked.connect(ConnectDca)
    exitbtn.clicked.connect(app.instance().exit)
    app.instance().exec_()
    set_radar.StopRadar()
    print('=======================================')


def model_predict():
    global realtime_data, model, data_mean, std, predict_result, text_label
    # print(realtime_data.empty())
    if not realtime_data.empty():
        # data_predict = realtime_data.get()
        # print(np.shape(data_predict))
        input_data = []
        raw_data = []
        # for b_f in range(2):
        #     rai_tmp = frame_buffer[b_f]
        #     input_data.append(rai_tmp)

        for f in range(32):
            # data = data_predict[f]
            data = realtime_data.get()
            # if f < 8:
            #     pass
            # else:
            #     if f == 40:
            #         break
            # Range Angle Image
            data = np.reshape(data, [-1, 4])
            data = data[:, 0:2:] + 1j * data[:, 2::]
            data = np.reshape(data, [16, 3, 4, 64])
            cdata1 = data[:, :, 0, :]
            cdata1 = np.transpose(cdata1, [0, 2, 1])
            cdata2 = data[:, :, 1, :]
            cdata2 = np.transpose(cdata2, [0, 2, 1])
            cdata3 = data[:, :, 2, :]
            cdata3 = np.transpose(cdata3, [0, 2, 1])
            cdata4 = data[:, :, 3, :]
            cdata4 = np.transpose(cdata4, [0, 2, 1])
            data_rai = np.concatenate((cdata1, cdata2, cdata3, cdata4), axis=2)
            raw_data.append(data_rai)
            rai_tmp = DSP_2t4r.Range_Angle(data_rai[:, :, 0:8], 1, [64, 64, 32])
            rai_tmp = rai_tmp.sum(0).T
            rai_tmp = rai_tmp[:, :32]
            rai_tmp = np.log10(rai_tmp) * 20
            rai_tmp = np.array(rai_tmp, dtype=np.float32)
            # plt.figure()
            # plt.imshow(rai_tmp)
            # plt.savefig('E:\\NTUT-master\\NTUT-Thesis\\DATA\\IWR6843_RawData\\realtime\\' + str(f) + '.png')
            # plt.close()
            input_data.append(rai_tmp)

        input_data = (input_data - data_mean) / std
        # print('Original shape: ', np.shape(input_data))
        input_data = np.reshape(input_data, [1, 32, 32, 32, 1])
        # print('Input shape :', np.shape(input_data))


        # save file
        np.save(output_path + text_label.toPlainText() + '_' + str(count) + '_test_data.npy', input_data)
        np.save(output_path + text_label.toPlainText() + '_' + str(count) + '_raw_data.npy', raw_data)

        pred = model.predict(input_data)
        label_cat = np.reshape(pred, [-1, 8])
        frame_predict = np.argmax(label_cat, axis=1)
        confidence_predict = []
        for z in range(32):
            # print(label_cat[z, frame_predict[z]])
            if (label_cat[z, frame_predict[z]]) > 0.8:
                confidence_predict.append(frame_predict[z])
                # print(frame_predict[z])

        seq_predict = np.argmax(np.bincount(frame_predict))
        print('Gesture is {}'.format(str(seq_predict + 1)))
        # print('*' * 20)
        print('Gesture is {}'.format(str(frame_predict + 1)) + 'Count:', len(frame_predict))
        # print('*' * 20)
        # print('Confidence 0.8 Gesture is {}'.format(str(np.array(confidence_predict) + 1)) + 'Count:', len(confidence_predict))
        # print('*' * 20)
        gesture_result.setText('Gesture:{}      True Label:'.format(seq_predict + 1))
        predict_result.append(frame_predict + 1)


if __name__ == '__main__':
    print('======Real Time Data Capture Tool======')
    output_path = 'E:/NTUT-master/NTUT-Thesis/DATA/IWR6843_RawData/result/real-time/velocity_offset/'
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    # ===============Select Model===============
    model_path = 'E:/NTUT-master/NTUT-Thesis/DATA/IWR6843_RawData/result/MODEL/2T4R_FoV70_BF_ratio_8_2/'
    model = load_model(model_path + 'model.h5')
    model.load_weights(model_path + '1_weights_0026.h5')
    data_mean = np.load(model_path + 'mean.npy')
    std = np.load(model_path + 'std.npy')
    # ==========================================
    true_label = 0
    count = 0
    previous_status = 0
    is_hand_there = 0
    frame_buffer = []
    predict_result = []
    # Queue for access data
    BinData = Queue()
    RDIData = Queue()
    RAIData = Queue()
    CAMData = Queue()
    CAMData2 = Queue()
    rawData = Queue()
    cam_rawData = Queue()
    cam_rawData2 = Queue()

    realtime_data = Queue()
    # Radar config
    frame_num_1 = 32
    adc_sample = 64
    chirp = 16
    tx_num = 3
    rx_num = 4
    radar_config = [adc_sample, chirp, tx_num, rx_num]
    frame_length = adc_sample * chirp * tx_num * rx_num * 2
    # Host setting
    address = ('192.168.33.30', 4098)
    buff_size = 2097152

    # # config DCA1000 to receive bin data
    # config_address = ('192.168.33.30', 4096)
    # FPGA_address_cfg = ('192.168.33.180', 4096)
    # cmd_order = ['9', 'E', '3', 'B', '5', '6']
    # sockConfig = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # sockConfig.bind(config_address)

    lock = threading.Lock()
    # cam1 = CamCapture(0, 'First', 0, lock, CAMData, cam_rawData, mode=1)
    # cam2 = CamCapture(0, 'Second', 0, lock, CAMData2, cam_rawData2, mode=1)

    # cls_pred = class_predict('Predict', realtime_data, model)
    collector = UdpListener('Listener', BinData, frame_length, address, buff_size, rawData, realtime_data, frame_num=frame_num_1)
    processor = DataProcessor('Processor', radar_config, BinData, RDIData, RAIData, realtime_data, 0, "0105", status=0)

    # pred_thread = threading.Thread(target=model_predict())
    # pred_thread.start()
    # cam1.start()
    # cam2.start()

    collector.start()
    processor.start()
    # cls_pred.start()
    plotIMAGE = threading.Thread(target=plot())
    plotIMAGE.start()

    # sockConfig.sendto(send_cmd('6'), FPGA_address_cfg)
    # sockConfig.close()
    collector.join(timeout=1)
    processor.join(timeout=1)
    # pred_thread.join(timeout=1)
    # cls_pred.join(timeout=1)
    # cam1.close()
    # cam2.close()

    with open(output_path + 'True_Label_' + true_label + '_result.csv', 'a', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['Label', true_label])
        writer.writerow(['Predict'])
        writer.writerows(predict_result)
    np.save(output_path + true_label + '_' + 'frame_predict.npy', predict_result)
    print("Program Close")
    sys.exit()
