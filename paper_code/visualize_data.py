import read_binfile
import numpy as np

data_path = 'C:/ti/mmwave_studio_02_00_00_02/mmWaveStudio/PostProc/'
file_name = 'adc_data_Raw_0.bin'

# radar config
frame = 64
sample = 64
chirp = 32
tx_num = 3
rx_num = 4
config = [frame, sample, chirp, tx_num, rx_num]
data = read_binfile.read_bin_file(data_path + file_name, config, 1, True, 4322)
