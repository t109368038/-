import threading as th
import numpy as np
import socket
import DSP_2t4r
import mmwave as mm
from mmwave.dsp.utils import Window


def run_proecss(raw_data,static_removal):
    frame_count = 0
    while True:

        range_resolution, bandwidth = mm.dsp.range_resolution(128)
        doppler_resolution = mm.dsp.doppler_resolution(bandwidth, 60, 33.02, 9.43, 16, 3)


        raw_data = np.reshape(raw_data,[-1,4,64])
        radar_cube = mm.dsp.range_processing(raw_data, window_type_1d=Window.HANNING)
        assert radar_cube.shape == (
            96, 4, 64), "[ERROR] Radar cube is not the correct shape!" #(numChirpsPerFrame, numRxAntennas, numADCSamples)

        # (3) Doppler Processing
        det_matrix, aoa_input = mm.dsp.doppler_processing(radar_cube, num_tx_antennas=3,
                                                          clutter_removal_enabled=static_removal,
                                                          window_type_2d=Window.HANNING, accumulate=True)

        det_matrix_vis = np.fft.fftshift(det_matrix, axes=1)


        # (4) Angle Processing sample, channel, chirp
        azimuth_ant_1 = aoa_input[:, :2 * 4, :]
        azimuth_ant_2 = aoa_input[:, 2 * 4:, :]
        elevation_ant_1 = aoa_input[:, 2, :]
        elevation_ant_2 = aoa_input[:, 8, :]
        elevation_combine = np.array([elevation_ant_1, elevation_ant_2]).transpose([1, 0, 2])

        # (4-1) Range Angle change to chirps, samples, channels
        azimuth_ant_1 = azimuth_ant_1.transpose([2, 0, 1])
        elevation_combine = elevation_combine.transpose([2, 0, 1])

        def Range_Angle(data, padding_size):
            rai_abs = np.fft.fft(data, n=padding_size, axis=2)
            rai_abs = np.fft.fftshift(np.abs(rai_abs), axes=2)
            rai_abs = np.flip(rai_abs, axis=1)
            return rai_abs

        azimuth_map = Range_Angle(azimuth_ant_1, 90)
        elevation_map = Range_Angle(elevation_combine, 90)

        # (5) Object Detection
        fft2d_sum = det_matrix.astype(np.int64)


        thresholdDoppler, noiseFloorDoppler = np.apply_along_axis(func1d=mm.dsp.ca_,
                                                                  axis=0,
                                                                  arr=fft2d_sum.T,
                                                                  l_bound=1.5,
                                                                  guard_len=2,
                                                                  noise_len=4)

        thresholdRange, noiseFloorRange = np.apply_along_axis(func1d=mm.dsp.ca_,
                                                              axis=0,
                                                              arr=fft2d_sum,
                                                              l_bound=2.5,
                                                              guard_len=2,
                                                              noise_len=4)

        thresholdDoppler, noiseFloorDoppler = thresholdDoppler.T, noiseFloorDoppler.T

        det_doppler_mask = (det_matrix > thresholdDoppler)
        det_range_mask = (det_matrix > thresholdRange)


        # Get indices of detected peaks
        full_mask = (det_doppler_mask & det_range_mask)
        det_peaks_indices = np.argwhere(full_mask == True)

        # peakVals and SNR calculation
        peakVals = fft2d_sum[det_peaks_indices[:, 0], det_peaks_indices[:, 1]]
        snr = peakVals - noiseFloorRange[det_peaks_indices[:, 0], det_peaks_indices[:, 1]]

        dtype_location = '(' + str(3) + ',)<f4' # 3 == numTxAntennas
        dtype_detObj2D = np.dtype({'names': ['rangeIdx', 'dopplerIdx', 'peakVal', 'location', 'SNR'],
                                   'formats': ['<i4', '<i4', '<f4', dtype_location, '<f4']})
        detObj2DRaw = np.zeros((det_peaks_indices.shape[0],), dtype=dtype_detObj2D)
        detObj2DRaw['rangeIdx'] = det_peaks_indices[:, 0].squeeze()
        detObj2DRaw['dopplerIdx'] = det_peaks_indices[:, 1].squeeze()
        detObj2DRaw['peakVal'] = peakVals.flatten()
        detObj2DRaw['SNR'] = snr.flatten()

        detObj2DRaw = mm.dsp.prune_to_peaks(detObj2DRaw, det_matrix, 32, reserve_neighbor=True) # 16 = numDopplerBins

        # --- Peak Grouping
        detObj2D = mm.dsp.peak_grouping_along_doppler(detObj2DRaw, det_matrix, 16) # 16 = numDopplerBins

        SNRThresholds2 = np.array([[2, 23], [10, 11.5], [35, 16]])
        peakValThresholds2 = np.array([[2, 275], [1, 400], [500, 0]])
        # SNRThresholds2 = np.array([[0, 15], [10, 16], [0 , 20]])
        # SNRThresholds2 = np.array([[0, 20], [10, 0], [0 , 0]])

        detObj2D = mm.dsp.range_based_pruning(detObj2D, SNRThresholds2, peakValThresholds2, 64, 0.5, # 64== numRangeBins
                                              range_resolution)

        azimuthInput = aoa_input[detObj2D['rangeIdx'], :, detObj2D['dopplerIdx']]
        # print(np.shape(detObj2D['dopplerIdx']))
        Psi, Theta, Ranges, velocity, xyzVec = mm.dsp.beamforming_naive_mixed_xyz(azimuthInput,
                                                                                  63-detObj2D['rangeIdx'],
                                                                                  detObj2D['dopplerIdx'],
                                                                                  range_resolution,
                                                                                  method='Bartlett')
        return  np.flip(det_matrix_vis),np.flip(azimuth_map.sum(0)),xyzVec


