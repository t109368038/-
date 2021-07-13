import numpy as np
import mediapipe as mp
from offline_process_3t4r_for_correct_for1443 import DataProcessor_offline
from training_data_preprocess.pixel2voxel import save_voxel,save_index_finger
from training_data_preprocess.meidipipe_run_by_vedio import save_mediapipe_point
from tools.read_binfile import read_bin_file

def radar_save_voxel(save_path,load_file_name, save_name):

    data = np.load(load_file_name, allow_pickle=True)
    out_radar_p = []
    out_center_p = []

    for i in range(len(data)):
        tmp_frame = data[i]
        arr = np.ndarray([1,25,25,25])
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

            if x<25 and y<25 and z<25 and x>0 and y>0 and z>0:
                # print("x:{} y:{} z:{}".format(x,y,z))
                arr[0][x][y][z] = True
                point_count += 1
                tx += tmp_frame[c][0]
                ty += tmp_frame[c][1]
                tz += tmp_frame[c][2]
        if point_count != 0 :
            out_center_p.append([tx/point_count, ty/point_count, tz/point_count])
        else:
            out_center_p.append([0,0,0])
        out_radar_p.append(arr)
    np.save(save_path+"out_voxel_" + str(save_name)+".npy", out_radar_p)
    # np.save(save_path+"out_center_p"+str(load_file_name)+".npy", out_center_p)
    print("voxel process Done")

class DataProcesser():
    def __init__(self,static=True):
        self.data_proecsss = DataProcessor_offline()
        self.Sure_staic_RM = static

    def load_raw_radar_data(self, input_data, save_path):
        self.save_path = save_path
        self.rawData = input_data
        self.frame_total_len = len(self.rawData)
        self.chirp = 32
        self.pd  = []

    def run(self,save_file_name):

        for i in range(self.frame_total_len):
            RDI, RAI, RAI_ele, PD = self.data_proecsss.run_proecss(self.rawData[i], \
                                0,self.Sure_staic_RM, self.chirp)
            self.pd.append(PD.T)
        np.save(self.save_path+"point_cloud"+str(save_file_name)+".npy",self.pd)
        print("   point cloud process Done")


if __name__ == '__main__':
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    name = "transfer_1"
    Gesture = ["circle", "eight", "rectangle", "up", "down", "left", "right"]
    # Gesture = ["circle"]
    # head_path = 'C:/Users/user/Desktop/thmouse_training_data/'
    head_path = 'D:\\Matt_yen_data\\NAS\\data\\bin_file_processed\\new_data_low_powered\\3t4r\\'
    save_path = 'D:/ForAndy/1443_point_cloud/'
    save_voxel_path = 'D:/ForAndy/1443_voxel/'
    static_romve = False
#------------ for single test ------------
    # # tmp_path = head_path + Gesture[6] + "/time" + str(3) + "/"
    # tmp_path = "C:/Users/user/Desktop/thmouse_training_data/"
    # print(tmp_path)
    # data_proecsss = DataProcesser(static_romve)
    # data_proecsss.load_raw_radar_data(file_path=tmp_path)
    # data_proecsss.run()
    # save_mediapipe_point(tmp_path, mp_hands)
    # save_voxel(tmp_path)
    # save_index_finger(tmp_path)

#------------ for loop test ------------
frame = 64
sample = 64
chirp = 32
tx_num = 3
rx_num = 4
config = [frame, sample, chirp, tx_num, rx_num]
for sense in range(0,2):
    for person in range(0,12):
        for gesture in range(0,12):
            for time in range(0,10):

                time = time + 1
                tmp_data_name = "adc_data_3t4r_"+str(sense)+"_"+str(person)+"_"+str(gesture)+"_00"+str(time)+"_process_0.bin"
                tmp_path = head_path + tmp_data_name
                # data_cube = read_bin_file(file_name=tmp_path, config=config, mode=0, header=False, packet_num=4322)
                # input_data = data_cube.transpose(0, 1, 3, 2)
                # input_data = np.reshape(input_data,[64,96,4,64])
                # print(input_data.shape)

                # data_proecsss = DataProcesser(static_romve)
                # data_proecsss.load_raw_radar_data(input_data= input_data , save_path=save_path)
                # data_proecsss.run(str(sense)+"_"+str(person)+"_"+str(gesture)+"_00"+str(time))

                pd_path = save_path + "point_cloud" + str(sense) + "_" + str(person) + "_" + str(gesture) + "_00" + str(time) + ".npy"
                radar_save_voxel(save_path=save_voxel_path, load_file_name=pd_path, save_name=str(sense)+"_"+str(person)+"_"+str(gesture)+"_00"+str(time))


