import numpy as np
import mediapipe as mp
from offline_process_3t4r_for_correct import DataProcessor_offline
from training_data_preprocess.pixel2voxel import save_voxel,save_index_finger
from training_data_preprocess.meidipipe_run_by_vedio import save_mediapipe_point


class DataProcesser():
    def __init__(self,static=True):
        self.data_proecsss = DataProcessor_offline()
        self.Sure_staic_RM = static

    def load_raw_radar_data(self, file_path, save_path):
        self.path = file_path
        self.save_path = save_path
        data = np.load(file_path+"raw.npy", allow_pickle=True)
        data = np.reshape(data, [-1, 4])
        data = data[:, 0:2:] + 1j * data[:, 2::]
        self.rawData = np.reshape(data, [-1, 48, 4, 64])
        self.frame_total_len = len(self.rawData)
        self.chirp = 16
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
    name = "moving_average3_out_mid"
    Gesture = ["circle", "eight", "rectangle", "up", "down", "left", "right"]
    head_path = 'C:/Users/user/Desktop/thmouse_training_data/'
    save_path = 'D:/thumouse_training_data/'+name+'/'
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

    for i in Gesture:
        for j in range(2,4):
            tmp_path = head_path + i + "/time" + str(j) + "/"
            savepath = save_path + i + "/time" + str(j) + "/"
            print(tmp_path)
            data_proecsss = DataProcesser(static_romve)
            data_proecsss.load_raw_radar_data(file_path=tmp_path, save_path=savepath)
            data_proecsss.run("_scr_"+ name)
            # save_mediapipe_point(tmp_path, savepath, mp_hands) # produce cam_hp.npy cam_hp1.npy
            save_voxel(tmp_path ,savepath,"_scr_"+ name)
            save_index_finger(tmp_path,savepath,"_scr_"+ name)  # out_cam_p.py

