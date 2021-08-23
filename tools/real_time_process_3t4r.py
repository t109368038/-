import threading as th
import numpy as np
import socket
import DSP_2t4r
import mmwave as mm
from mmwave.dsp.utils import Window
import arlpy as ar
from tensorflow.keras.models import Sequential, load_model
import tensorflow.keras.optimizers as optimizers
from tensorflow.keras.callbacks import LearningRateScheduler


class UdpListener(th.Thread):
    def __init__(self, name, bin_data, data_frame_length, data_address, buff_size, save_data, realtime_data, record_status=0):
        """
        :param name: str
                        Object name

        :param bin_data: queue object
                        A queue used to store adc data from udp stream

        :param data_frame_length: int
                        Length of a single frame

        :param data_address: (str, int)
                        Address for binding udp stream, str for host IP address, int for host data port

        :param buff_size: int
                        Socket buffer size
        """
        th.Thread.__init__(self, name=name)

        self.bin_data = bin_data
        self.frame_length = data_frame_length
        self.data_address = data_address
        self.buff_size = buff_size
        self.save_data = save_data
        self.realtime_data = realtime_data
        self.status = 0
        self.record_status = record_status

    def run(self):
        # convert bytes to data type int16
        dt = np.dtype(np.int16)
        dt = dt.newbyteorder('<')
        # array for putting raw data
        np_data = []
        # count frame
        self.count_frame = 0
        data_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        data_socket.bind(self.data_address)
        print("Create Data Socket Successfully")
        print("Waiting For The Data Stream")
        print('=======================================')
        real_time_tmp = []
        # main loop
        while True:
            data, addr = data_socket.recvfrom(self.buff_size)
            data = data[10:]
            np_data.extend(np.frombuffer(data, dtype=dt))
            # while np_data length exceeds frame length, do following

            if len(np_data) >= self.frame_length:
                # print(self.frame_length)
                # print("Frame No.", count_frame)
                # put one frame data into bin data array
                if self.status == 1:
                    # print("Frame No.", self.count_frame)

                    if self.count_frame < 32:
                        self.save_data.put(np_data[0:self.frame_length])
                        self.realtime_data.put(np_data[0:self.frame_length])
                        self.count_frame += 1
                    else:
                        # self.realtime_data.put(real_time_tmp)
                        # real_time_tmp = []
                        print('Collected')
                        self.record_status = 1
                        self.status = 0
                        self.count_frame = 0
                self.bin_data.put(np_data[0:self.frame_length])
                # remove one frame length data from array
                np_data = np_data[self.frame_length:]


class DataProcessor(th.Thread):
    def __init__(self, name, config, bin_queue, rdi_queue, rai_queue, realtime_data, raw_queue=None, file_name=0,
                 status=0, hand_status=0):
        """
        :param name: str
                        Object name

        :param config: sequence of ints
                        Radar config in the order
                        [0]: samples number
                        [1]: chirps number
                        [3]: transmit antenna number
                        [4]: receive antenna number

        :param bin_queue: queue object
                        A queue for access data received by UdpListener

        :param rdi_queue: queue object
                        A queue for store RDI

        :param rai_queue: queue object
                        A queue for store RDI

        """
        th.Thread.__init__(self, name=name)
        self.adc_sample = config[0]
        self.chirp_num = config[1]
        self.tx_num = config[2]
        self.rx_num = config[3]
        self.bin_queue = bin_queue
        self.rdi_queue = rdi_queue
        self.rai_queue = rai_queue
        self.raw_queue = raw_queue
        self.realtime_data = realtime_data

        self.filename = file_name
        self.status = status
        self.weight_matrix = np.zeros([181, 8], dtype=complex)
        self.out_matrix = np.zeros([8192, 181], dtype=complex)

        self.previous_status = 0
        self.is_hand_there = hand_status

        Fc = 77.2e9
        count = 0
        lambda_start = 3e8 / Fc
        for theta in range(-90, 91):
            d = 0.5 * lambda_start * np.sin(theta * np.pi / 180)
            beamforming_factor = np.array([0, d, 2 * d, 3 * d, 4 * d, 5 * d, 6 * d, 7 * d]) / (3e8 / Fc)
            self.weight_matrix[count, :] = np.exp(-1j * 2 * np.pi * beamforming_factor)
            count += 1

    def run(self):
        frame_count = 0
        while True:
            # if not self.realtime_data.empty():
                data = self.bin_queue.get()
                # data = self.realtime_data.get()
                data = np.reshape(data, [-1, 4])
                data = data[:, 0:2:] + 1j * data[:, 2::]
                # frame = np.copy(data)
                # data = np.reshape(data, [self.chirp_num * self.tx_num, 4, self.adc_sample])
                # data = data.transpose([0, 2, 1])
                # ch1_data = data[0:self.chirp_num * self.tx_num:3, :, :]
                # ch3_data = data[1:self.chirp_num * self.tx_num:3, :, :]
                # ch2_data = data[2:self.chirp_num * self.tx_num:3, :, :]
                # data_rai = np.concatenate([ch1_data, ch3_data, ch2_data], axis=2)
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
                # print(np.shape(data_rai))

                rai_tmp = DSP_2t4r.Range_Angle(data_rai[:, :, 0:8], 1, [64, 64, 32])
                rai_tmp = rai_tmp.sum(0).T
                rai_tmp = rai_tmp[:, :32]
                rai_tmp = np.log10(rai_tmp) * 20
                rai_tmp = np.array(rai_tmp, dtype=np.float32)

                # print('Previous RAI mean: ', self.previous_status)
                # print('Current RAI mean: ', np.mean(rai_tmp))

                # print('Difference:', np.mean(rai_tmp) - self.previous_status)

                # print(((np.mean(rai_tmp) - self.previous_status) > 0))
                # print(((np.mean(rai_tmp) - self.previous_status) < 20))

                # if ((np.mean(rai_tmp) - self.previous_status) > 0.9) and ((np.mean(rai_tmp) - self.previous_status) < 20):
                #     if self.is_hand_there == 0:
                #         print('There is a hand')
                #         self.is_hand_there = 1
                # # else:
                #     # print('No hands')
                # self.previous_status = np.mean(rai_tmp)

                # print(np.shape(rai_tmp))
                # rai_tmp = np.array(rai_tmp, dtype=np.float32)
                self.rdi_queue.put(rai_tmp)
                # frame_count += 1
                # rdi = DSP_2t4r.Range_Doppler(ch2_data, mode=1, padding_size=[128, 64])
                # # rai = DSP_2t4r.Range_Angle(data[:, :, 0:4], mode=1, padding_size=[128, 64, 64])
                # self.rdi_queue.put(rdi)
                # # self.rai_queue.put(rai)
                #
                # range_resolution, bandwidth = mm.dsp.range_resolution(64)
                # doppler_resolution = mm.dsp.doppler_resolution(bandwidth)

                # raw_data = self.bin_queue.get()
                # data = np.reshape(raw_data, [-1, 4])
                # frame = data[:, 0:2:] + 1j * data[:, 2::]
                # frame = frame.reshape(self.chirp_num * self.tx_num, self.rx_num, self.adc_sample)
                #
                # radar_cube = mm.dsp.range_processing(frame, window_type_1d=Window.HANNING)
                #
                # assert radar_cube.shape == (
                #     3*16, 4, 64), "[ERROR] Radar cube is not the correct shape!"
                #
                # det_matrix, aoa_input = mm.dsp.doppler_processing(radar_cube, num_tx_antennas=3,
                #                                                   clutter_removal_enabled=False)

                # # (4) Object Detection
                # fft2d_sum = det_matrix.astype(np.int64)
                # thresholdDoppler, noiseFloorDoppler = np.apply_along_axis(func1d=mm.dsp.ca_,
                #                                                           axis=0,
                #                                                           arr=fft2d_sum.T,
                #                                                           l_bound=1.5,
                #                                                           guard_len=4,
                #                                                           noise_len=16)
                #
                # thresholdRange, noiseFloorRange = np.apply_along_axis(func1d=mm.dsp.ca_,
                #                                                       axis=0,
                #                                                       arr=fft2d_sum,
                #                                                       l_bound=2.5,
                #                                                       guard_len=4,
                #                                                       noise_len=16)
                #
                # thresholdDoppler, noiseFloorDoppler = thresholdDoppler.T, noiseFloorDoppler.T
                # det_doppler_mask = (det_matrix > thresholdDoppler)
                # det_range_mask = (det_matrix > thresholdRange)
                # # Get indices of detected peaks
                # full_mask = (det_doppler_mask & det_range_mask)
                # det_peaks_indices = np.argwhere(full_mask == True)
                #
                # # peakVals and SNR calculation
                # peakVals = fft2d_sum[det_peaks_indices[:, 0], det_peaks_indices[:, 1]]
                # snr = peakVals - noiseFloorRange[det_peaks_indices[:, 0], det_peaks_indices[:, 1]]
                #
                # dtype_location = '(' + str(3) + ',)<f4'
                # dtype_detObj2D = np.dtype({'names': ['rangeIdx', 'dopplerIdx', 'peakVal', 'location', 'SNR'],
                #                            'formats': ['<i4', '<i4', '<f4', dtype_location, '<f4']})
                # detObj2DRaw = np.zeros((det_peaks_indices.shape[0],), dtype=dtype_detObj2D)
                # detObj2DRaw['rangeIdx'] = det_peaks_indices[:, 0].squeeze()
                # detObj2DRaw['dopplerIdx'] = det_peaks_indices[:, 1].squeeze()
                # detObj2DRaw['peakVal'] = peakVals.flatten()
                # detObj2DRaw['SNR'] = snr.flatten()
                #
                #
                # # Further peak pruning. This increases the point cloud density but helps avoid having too many detections around one object.
                # detObj2DRaw = mm.dsp.prune_to_peaks(detObj2DRaw, det_matrix, 16, reserve_neighbor=True)
                #
                # # --- Peak Grouping
                # detObj2D = mm.dsp.peak_grouping_along_doppler(detObj2DRaw, det_matrix, 16)
                # SNRThresholds2 = np.array([[2, 23], [10, 11.5], [35, 16.0]])
                # peakValThresholds2 = np.array([[4, 275], [1, 400], [500, 0]])
                # detObj2D = mm.dsp.range_based_pruning(detObj2D, SNRThresholds2, peakValThresholds2, 64, 32, range_resolution)
                # # print(detObj2D)
                #
                # aa = det_matrix
                # # print(np.shape(aa))

                # # print(detObj2D[0:2])
                # range_index = detObj2D['rangeIdx']
                # doppler_index = detObj2D['dopplerIdx']
                # # print(range_doppler_index)
                # # if len(detObj2D) != 0:
                # #     # print(len(detObj2D))
                # #     for t in range(len(np.shape(detObj2D))):
                # #         aa[range_index[t], doppler_index[t]] = 250
                # self.rdi_queue.put(aa)
                #
                # antenna = 8
                # antenna_steer = 4
                # d = 3e8 / 60e9
                # pos = np.arange(antenna_steer) * d * 0.5
                # ang = 32  # Beamforming point
                # theta = np.linspace(-70, 70, num=ang, endpoint=True)
                # theta = np.deg2rad(theta)
                # steering = ar.bf.steering_plane_wave(pos, 3e8, theta)

                # aoa_input = np.transpose(aoa_input, [2, 0, 1])
                # aoa_input = aoa_input[:, :, 0:antenna]
                #
                # rai_tmp = np.zeros([aoa_input.shape[0], aoa_input.shape[1], 32])
                # for i in range(aoa_input.shape[0]):
                #     # for j in range(aoa_input.shape[1]):
                #     rai_tmp[i, :, :] = ar.bf.bartlett(aoa_input[i, :, :].T, 60e9, steering,
                #                                       complex_output=False)
                # rai_tmp = np.log10(np.abs(rai_tmp).sum(0)[32:, :]) * 20
                # rai_tmp = np.log10(np.abs(rai_tmp).sum(0)) * 20

                # aoa_input = np.transpose(radar_cube, [2, 0, 1])[:, :, 0:antenna]

                # azimuthInput = aoa_input[detObj2D['rangeIdx'], :, detObj2D['dopplerIdx']]
                # # print(azimuthInput)
                # x, y, z = mm.dsp.naive_xyz(azimuthInput.T)
                # xyzVecN = np.zeros((3, x.shape[0]))
                # xyzVecN[0] = x * range_resolution * detObj2D['rangeIdx']
                # xyzVecN[1] = y * range_resolution * detObj2D['rangeIdx']
                # xyzVecN[2] = z * range_resolution * detObj2D['rangeIdx']
                # Psi, Theta, Ranges, velocity, xyzVec = mm.dsp.beamforming_naive_mixed_xyz(azimuthInput, 63 - detObj2D['rangeIdx'], detObj2D['dopplerIdx'],
                #                                                                 range_resolution, method='Bartlett')
                # self.rai_queue.put(xyzVec) #Output Point Cloud

                # # (5) 3D-Clustering
                # # detObj2D must be fully populated and completely accurate right here
                # numDetObjs = detObj2D.shape[0]
                # dtf = np.dtype({'names': ['rangeIdx', 'dopplerIdx', 'peakVal', 'location', 'SNR'],
                #                 'formats': ['<f4', '<f4', '<f4', dtype_location, '<f4']})
                # detObj2D_f = detObj2D.astype(dtf)
                # detObj2D_f = detObj2D_f.view(np.float32).reshape(-1, 7)
                #
                # # Fully populate detObj2D_f with correct info
                # for i, currRange in enumerate(Ranges):
                #     if i >= (detObj2D_f.shape[0]):
                #         # copy last row
                #         detObj2D_f = np.insert(detObj2D_f, i, detObj2D_f[i - 1], axis=0)
                #     if currRange == detObj2D_f[i][0]:
                #         detObj2D_f[i][3] = xyzVec[0][i]
                #         detObj2D_f[i][4] = xyzVec[1][i]
                #         detObj2D_f[i][5] = xyzVec[2][i]
                #     else:  # Copy then populate
                #         detObj2D_f = np.insert(detObj2D_f, i, detObj2D_f[i - 1], axis=0)
                #         detObj2D_f[i][3] = xyzVec[0][i]
                #         detObj2D_f[i][4] = xyzVec[1][i]
                #         detObj2D_f[i][5] = xyzVec[2][i]
                #
                #         # radar_dbscan(epsilon, vfactor, weight, numPoints)
                # #        cluster = radar_dbscan(detObj2D_f, 1.7, 3.0, 1.69 * 1.7, 3, useElevation=True)
                # if len(detObj2D_f) > 0:
                #     cluster = mm.clustering.radar_dbscan(detObj2D_f, 0, doppler_resolution, use_elevation=True)
                #
                #     cluster_np = np.array(cluster['size']).flatten()
                #     if cluster_np.size != 0:
                #         if max(cluster_np) > max_size:
                #             max_size = max(cluster_np)


class class_predict(th.Thread):
    def __init__(self, name, rai_data, model):
        """
        """
        th.Thread.__init__(self, name=name)
        self.rai = rai_data
        self.model = model

    def run(self):
        data = []
        while True:

            if not self.rai.empty():
                frame = self.rai.get()
                data.append(frame)
                print(np.shape(data))

                # print(np.shape(data))
                # data = np.reshape(data, [1, 32, 32, 32, 1])
                # pred = self.model.predict(data)
                # print(np.shape(pred))
                # # pred = pred.sum(0)
                # # pred = pred.sum(0)
                #
                # label_cat = np.reshape(pred, [-1, 8])
                # cls = np.argmax(label_cat, axis=1)
                #
                # # print('Gesture is {}'.format(str(cls + 1)))
                # print(cls + 1)
