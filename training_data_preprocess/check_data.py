import  numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def show_camera_hand(path):
    tmp_path = path
    cam = True

    if cam :
        cam_voxel = np.load(tmp_path + "out_cam_p.npy", allow_pickle=True)
        print("cam_voxl: {} ".format(np.shape(cam_voxel)))

    radar_voxel = np.load(tmp_path + "point_cloud.npy", allow_pickle=True)
    print("radar_voxl: {} ".format(np.shape(radar_voxel[0])))
    print("radar_voxl: {} ".format(radar_voxel[0]))
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    nrx = []
    nry = []
    nrz = []
    nrx1 = []
    nry1 = []
    nrz1= []
    mark = "."
    for i in range(len(radar_voxel)):
        ax.clear()
        if cam:
            x = cam_voxel[i,0,:]
            y = cam_voxel[i,1,:]
            z = cam_voxel[i,2,:]
        rx = []
        ry = []
        rz = []
        low_rx = []
        low_ry = []
        low_rz = []
        if radar_voxel[i] != [] :
            tmp = radar_voxel[i]
            for j in range(len(tmp)):
                if tmp[j,3] > 55:
                    rx.append(tmp[j,0])
                    ry.append(tmp[j,1])
                    rz.append(tmp[j,2])
                else:
                    low_rx.append(tmp[j, 0])
                    low_ry.append(tmp[j, 1])
                    low_rz.append(tmp[j, 2])

        rx = np.mean(rx,axis=0)
        ry = np.mean(ry,axis=0)
        rz = np.mean(rz,axis=0)
        low_rx = np.mean(low_rx,axis=0)
        low_ry = np.mean(low_ry,axis=0)
        low_rz = np.mean(low_rz,axis=0)
        nrx.append(rx)
        nry.append(ry)
        nrz.append(rz)
        nrx1.append(low_rx)
        nry1.append(low_ry)
        nrz1.append(low_rz)

        if len(nrx) > 2:
            nrx = nrx[1:]
            nry = nry[1:]
            nrz = nrz[1:]
            nrx1 = nrx1[1:]
            nry1 = nry1[1:]
            nrz1 = nrz1[1:]
        ax.plot3D(nrx, nry, nrz, marker="8", color="red")
        ax.plot3D(nrx1, nry1, nrz1, marker="8", color="green")

        scale_axis = 0.45
        ax.set_zlim3d(bottom=-1 * scale_axis, top=scale_axis)
        ax.set_ylim(bottom=0, top=scale_axis)
        ax.set_xlim(left=-1 * scale_axis, right=scale_axis)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        if cam:
            ax.plot3D(x, y, z, marker=mark, color="gray")

        plt.draw()
        plt.pause(0.006)


if __name__ == '__main__':


    Gesture = ["circle", "eight", "rectangle", "up", "down", "left", "right"]
    head_path = 'C:/Users/user/Desktop/thmouse_training_data/'
    static_romve = True
    tmp_path = head_path + Gesture[4]+ "/time" + str(2) + "/"
    # tmp_path = 'C:/Users/user/Desktop/thmouse_training_data/'
    #
    show_camera_hand(tmp_path)


    # for i in Gesture:
    #     for j in range(2):
            # tmp_path = head_path + i + "/time" + str(j) + "/"
            # print(tmp_path)


    # show_camera_hand(tmp_path)

    # pd_path = tmp_path + 'point_cloud.npy'

    # pd = np.load(pd_path,allow_pickle=True)
    #
    # for i in range(1000):
    #     print("============< {} >============".format(i))
        # tmppd = pd[i]