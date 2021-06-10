import numpy as np
import mediapipe as mp
from offline_process_3t4r_for_correct import DataProcessor_offline
from training_data_preprocess.meidipipe_run_by_vedio import save_mediapipe_point
from training_data_preprocess.pixel2voxel import save_voxel,save_index_finger

class DataProcesser():
    def __init__(self,static=True):
        self.data_proecsss = DataProcessor_offline()
        self.Sure_staic_RM = static

    def load_raw_radar_data(self, file_path):
        self.path = file_path
        data = np.load(file_path+"raw.npy", allow_pickle=True)
        data = np.reshape(data, [-1, 4])
        data = data[:, 0:2:] + 1j * data[:, 2::]
        self.rawData = np.reshape(data, [-1, 48, 4, 64])
        self.frame_total_len = len(self.rawData)
        self.chirp = 16
        self.pd  = []

    def run(self):
        for i in range(self.frame_total_len):
            RDI, RAI, RAI_ele, PD = self.data_proecsss.run_proecss(self.rawData[i], \
                                0,self.Sure_staic_RM, self.chirp)
            self.pd.append(PD.T)
        np.save(self.path+"point_cloud.npy",self.pd)
        print("   point cloud process Done")


if __name__ == '__main__':
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    Gesture = ["circle", "eight", "rectangle", "up", "down", "left", "right"]
    head_path = 'C:/Users/user/Desktop/thmouse_training_data/'
    static_romve = True
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

    # for i in Gesture:
    #     for j in range(1,4):
    #         tmp_path = head_path + i + "/time" + str(j) + "/"
    #         print(tmp_path)
    #         data_proecsss = DataProcesser(static_romve)
    #         data_proecsss.load_raw_radar_data(file_path = tmp_path)
    #         data_proecsss.run()
    #         save_mediapipe_point(tmp_path, mp_hands)
    #         save_voxel(tmp_path)
    #         save_index_finger(tmp_path)
tmp_path = 'C:/Users/user/Desktop/thmouse_training_data/'
data_proecsss = DataProcesser(static_romve)
data_proecsss.load_raw_radar_data(file_path = tmp_path)
data_proecsss.run()
save_mediapipe_point(tmp_path, mp_hands)
save_voxel(tmp_path)
save_index_finger(tmp_path)