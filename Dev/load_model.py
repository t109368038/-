import tensorflow as tf
from tensorflow.keras.models import load_model


# model = load_model('E:\pycharm_project\Andy-is-very-good\Dev/thm_model_2.h5')
model = load_model('E:/mmWave/save_model/1t4r_no_log_square_RAI_model_v1.h5')

model.summary()

