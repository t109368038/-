import sys

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import mmwave.clustering as clu
import mmwave.dsp as dsp
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import DSP_2t4r
import os
def plot_rdi(rdi):
    plt.title('Heatmap of 2D normally distributed data points')
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    #
    # plt.hist2d(x, y, bins=N_bins, normed=False, cmap='plasma')


def plot_pd(pos):
    ax = fig.add_subplot(111, projection='3d')
    ax.set_zlim3d(bottom=-5, top=5)
    ax.set_ylim(bottom=0, top=10)
    ax.set_xlim(left=-4, right=4)
    ax.view_init(elev=-174, azim=-90)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2])
    ax.scatter(0,0,0,c='red')
    # plt.show()
    # return None

if __name__ == '__main__':
    PDdata = np.load("D:/kaiku_report/20210414/pd.npy",allow_pickle=True)
    rdi = np.load("D:/kaiku_report/20210414/rdi.npy",allow_pickle=True)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    # plot_pd(PDdata[0])
    # plt.show()

    plt.ion()
    for i in range(len(PDdata)):
        print(i)
        plot_pd(PDdata[i])
        plt.draw()
        plt.pause(0.0001)
        plt.clf()
