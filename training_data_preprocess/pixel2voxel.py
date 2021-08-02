import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting


def load_gt(path):
    # path = 'C:/Users/user/Desktop/thmouse_training_data/'
    hand_pd_1 = np.load(path + "cam_hp1.npy", allow_pickle=True)
    print("self.hand_pd_1 len is {}".format(hand_pd_1.shape))
    hand_pd_2 = np.load(path + "cam_hp.npy", allow_pickle=True)
    print("self.hand_pd_2 len is {}".format(hand_pd_2.shape))

    return  hand_pd_1,hand_pd_2


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
    x2 += 0.015

    y2 = np.where(y2 is not None, y2, np.double(480))
    y2 = y2.astype(np.double)
    y2 = (y2 * scale_y2)
    y2 = (y2 * -1) + ((480 * scale_y2))
    y2 = np.round(y2, 3)

    return x2 * -1, y2, y1 * -1 #22cm


def show_voxel(path,cam_path):

    fig = plt.figure()
    data = np.load(path+"point_cloud_scr_transfer_0dot5.npy",allow_pickle=True)
    hand_pd_1, hand_pd_2 = load_gt(cam_path)
    print(hand_pd_2.shape)
    out_radar_p = []
    out_cam_p = []
    # sure_raw = True
    sure_raw = False
    for i in range(len(data)):
        plt.cla()
        tmp_frame = data[i]
        arr = np.ndarray([25,25,25])
        arr[:] = False
        colors = np.empty([25,25,25], dtype=object)
        empty = True
        ##===========image ground thurth=========
        X, Y, Z = GetGroundTruth(hand_pd_1[i * 2, :], hand_pd_1[i * 2 + 1, :],
                                 hand_pd_2[i * 2, :], hand_pd_2[i * 2 + 1, :])

        if sure_raw == True:
            X = X[8]/0.015
            Y = Y[8]/0.015
            Z = Z[8]/0.015
        else:
            X = int(np.round(X[8] / 0.015 + 12.5))
            Y = int(np.round(Y[8] / 0.015 - 10))
            Z = int(np.round(Z[8] / 0.015 + 12.5))
            print(X, Y, Z)

            arr[X][Y][Z] = True
            colors[X][Y][Z] = "green"
        ##========================================
        point_count = 0
        px = 0; pz = 0; py = 0;
        for c in range(tmp_frame.shape[0]):
            if sure_raw == True:
                x = tmp_frame[c][0]
                y = tmp_frame[c][1]
                z = tmp_frame[c][2]
                # print("x:{} y:{} z:{}".format(x, y, z))

                px += x
                py += y
                pz += z
                point_count += 1
                empty = False

            else:
                # print("x:{} y:{} z:{}".format(tmp_frame[c][0] / 0.009 , tmp_frame[c][1] / 0.009 , tmp_frame[c][2] / 0.009 ))

                x = round(tmp_frame[c][0] / 0.015 + 12.5)
                # x = round(tmp_frame[c][0] / 0.015 )
                y = round(tmp_frame[c][1] / 0.015 )
                # z = round(tmp_frame[c][2] / 0.015 )
                z = round(tmp_frame[c][2] / 0.015 + 12.5)

                empty = False

                if x<25 and y<25 and z<25 and x>0 and y>0 and z>0:
                    px += x ; py += y  ; pz +=z
                    print("x:{} y:{} z:{}".format(x,y,z))
                    arr[x][y][z] = True
                    colors[x][y][z] = "red"
                    point_count += 1
        if empty:
            continue
        else:
            if sure_raw == False:
                #=== set the radar center
                colors[12][0][0] = 'blue'
                arr[12][0][0] = True
                ax = fig.add_subplot(projection='3d')
                ax.view_init(elev=33., azim=45)
                # ax.view_init(elev=78, azim=90)
                ax.voxels(arr, facecolors=colors, edgecolor='k')
                plt.draw()
                # plt.pause(0.001)

                plt.pause(0.5)

        print("=============================")

        if point_count != 0:
            px = px / point_count
            py = py / point_count
            pz = pz / point_count
        out_cam_p.append([X,Y,Z])
        out_radar_p.append([px,py,pz])

    # np.save('C:/Users/user/Desktop/thmouse_training_data/out_cam_p.npy', out_cam_p)
    # np.save('C:/Users/user/Desktop/thmouse_training_data/out_radar_p.npy', out_radar_p)

def save_voxel(head_path, save_path,load_file_name):
    data = np.load(save_path + "point_cloud"+ str(load_file_name) +".npy",allow_pickle=True)
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
    np.save(save_path+"out_radar_p" +str(load_file_name)+".npy", out_radar_p)
    np.save(save_path+"out_center_p"+str(load_file_name)+".npy", out_center_p)
    print("   voxel  process Done")

def save_index_finger(path,save_path,savename):
    hand_pd_1, hand_pd_2 = load_gt(path)

    out_cam_p = []
    ##===========image ground thurth=========
    cc = int(len(hand_pd_1) / 2)
    try:

        for i in range(cc):
                X, Y, Z = GetGroundTruth(hand_pd_1[i * 2, :], hand_pd_1[i * 2 + 1, :],
                                     hand_pd_2[i * 2, :], hand_pd_2[i * 2 + 1, :])
                out_cam_p.append([X,Y,Z])
    except Exception  as e :
        print("ErrorRRRRRRRRRRR")
        print("-------------------------")
        print(e)
    np.save(save_path + "out_cam_p"+str(savename)+".npy", out_cam_p)

    ##========================================

def show_error():
    out_cam_p = np.load('C:/Users/user/Desktop/thmouse_training_data/out_cam_p.npy', allow_pickle=True)
    out_radar_p = np.load('C:/Users/user/Desktop/thmouse_training_data/out_radar_p.npy', allow_pickle=True)
    show_len(out_cam_p)
    show_len(out_radar_p)
    total_len = range(len(out_cam_p))
    # fig = plt.figure()
    plt.subplot(3, 1, 1)
    tmp_1 = np.mean(out_cam_p)
    tmp_2 = np.mean(out_radar_p)
    tmp  = tmp_1 - tmp_2
    out_cam_p += tmp
    plt.plot(total_len, out_cam_p[:,0], label="camera")
    plt.plot(total_len, out_radar_p[:,0], label="radar")
    plt.title('ALL frame on X axis', color='blue')
    plt.legend()

    plt.subplot(3, 1, 2)
    tmp_1 = np.mean(out_cam_p)
    tmp_2 = np.mean(out_radar_p)
    tmp  = tmp_1 - tmp_2
    plt.plot(total_len, out_cam_p[:,1], label="camera")
    plt.plot(total_len, out_radar_p[:,1], label="radar")
    plt.title('ALL frame on Y axis', color='blue')
    plt.legend()

    plt.subplot(3, 1, 3)
    tmp_1 = np.mean(out_cam_p)
    tmp_2 = np.mean(out_radar_p)
    tmp  = tmp_1 - tmp_2
    plt.plot(total_len, out_cam_p[:,2], label="camera")
    plt.plot(total_len, out_radar_p[:,2], label="radar")
    plt.legend()
    plt.title('ALL frame on Z axis', color='blue')
    plt.show()


def show_len(x):
    print("len is :{}".format(len(x)))
if __name__ == '__main__':
    # path = "C:/Users/user/Desktop/thmouse_training_data/circle/time2/"
    # cam_path = "C:/Users/user/Desktop/thmouse_training_data/right/time2/"
    # path = "D:/thumouse_training_data/transfer_0dot2/circle/time2/"
    path = "D:/thumouse_new_dataset_32_32/transfer_0dot5/circle/time3/"
    cam_path = "D:/thumouse_new_dataset_32_32/transfer_0dot5/circle/time3/"
    show_voxel(path,cam_path)
    # show_error()
    # save_voxel(path)