import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.utils import to_categorical

def data_normalize(datainput):
    datanum = datainput.shape[0]
    channel_num = datainput.shape[4]
    tempnom = np.zeros(np.shape(datainput))
    for j in range(channel_num):
        for i in range(datanum):
            tempdata0 = datainput[i, :, :, :, j]
            tepmax = tempdata0.max()
            tepmin = tempdata0.min()
            tempmaxmin = tepmax - tepmin
            temp = tempdata0 - tepmin
            temp = temp / tempmaxmin
            tempnom[i, :, :, :, j] = temp
        return tempnom

def data_normalize_channel(datainput):
    datanum = datainput.shape[0]
    channel_num = datainput.shape[4]
    tempnom = np.zeros(np.shape(datainput))
    for j in range(channel_num):
        # for i in range(datanum):
        tempdata0 = datainput[:, :, :, :, j]
        tepmax = tempdata0.max()
        tepmin = tempdata0.min()
        tempmaxmin = tepmax - tepmin
        temp = tempdata0 - tepmin
        temp = temp / tempmaxmin
        tempnom[:, :, :, :, j] = temp
    return tempnom


def data_normalize_toall(datainput):
    tempdata0 = np.reshape(datainput, -1)
    tepmax = tempdata0.max()
    tepmin = tempdata0.min()
    tempmaxmin = tepmax - tepmin
    temp = tempdata0 - tepmin
    temp = temp / tempmaxmin
    temp1 = np.reshape(temp,
                       [np.shape(datainput)[0], np.shape(datainput)[1], np.shape(datainput)[2], np.shape(datainput)[3],
                        np.shape(datainput)[4]])
    return temp1


def data_Standard(datainput):
    datanum = datainput.shape[0]
    channel_num = datainput.shape[4]
    tempnom = np.zeros(np.shape(datainput))
    for j in range(channel_num):
        for i in range(datanum):
            tempdata0 = datainput[i, :, :, :, j]
            tepmean = tempdata0.mean()
            tempstd = tempdata0.std()
            temp = tempdata0 - tepmean
            temp = temp / tempstd
            tempnom[i, :, :, :, j] = temp
        return tempnom


def data_Standard_toall(datainput):
    tempdata0 = np.reshape(datainput, -1)
    tepmean = tempdata0.mean()
    tempstd = tempdata0.std()
    temp = tempdata0 - tepmean
    temp = temp / tempstd
    temp1 = np.reshape(temp,
                       [np.shape(datainput)[0], np.shape(datainput)[1], np.shape(datainput)[2], np.shape(datainput)[3],
                        np.shape(datainput)[4]])
    return temp1


def data_Standardization(data):
    num_train = data.shape[0]
    width_train = data.shape[2]
    height_train = data.shape[3]
    channel_train = data.shape[4]
    x_train_mean_perChannel = np.zeros((num_train, 64, width_train, height_train, channel_train))

    for i in range(0, channel_train):
        temp = np.mean(data[:, :, :, :, i])
        temp = np.repeat(temp, height_train, axis=0)
        temp_2 = temp[np.newaxis, ...]
        temp_2 = np.repeat(temp_2, width_train, axis=0)
        temp_2 = temp_2[np.newaxis, ...]
        temp_2 = np.repeat(temp_2, 64, axis=0)
        temp_3 = temp_2[np.newaxis, ...]
        temp_3 = np.repeat(temp_3, num_train, axis=0)
        x_train_mean_perChannel[:, :, :, :, i] = temp_3
        # print(i)

    x_train_std = np.zeros((1, channel_train))
    for j in range(0, channel_train):
        x_train_std[:, j] = np.std(data[:, :, :, :, j])

    data_s0_1 = (data - x_train_mean_perChannel) / x_train_std

    return data_s0_1


def mattyen_rand(x_train, x_test, y_train, y_test):
    x_train_new = x_train
    y_train_new = y_train
    x_test_new = x_test
    y_test_new = y_test

    for i in range(0, 1440):
        # x_train_mean = np.mean(x_train_new[i, :, :, :, :])
        # x_test_mean = np.mean(x_test_new[is, :, :, :, :])
        # x_train_new[i, :, :, :, :] -= x_train_mean
        # x_test_new[i, :, :, :, :]  -= x_test_mean
        if i % 10 == 5 or i % 10 ==6  or i % 10 == 7 or i % 10 == 8 or i % 10 == 9:
            #    if i % 10 == 9:
            x_train_new[i] = x_test[i]
            y_train_new[i] = y_test[i]
        # if i % 10 == 0 or i % 10 == 6 or i % 10 == 7 or i % 10 == 8 or i % 10 == 4:
            x_test_new[i] = x_train[i]
            y_test_new[i] = y_train[i]

    return x_train_new, x_test_new, y_train_new, y_test_new
def plot_confusion_matrix(cm_normalized, classes, save_file=False):
    plt.figure(figsize=(12, 8), dpi=60)
    x_location = np.array(range(classes))
    x, y = np.meshgrid(x_location, x_location)

    np1 = np.zeros([12])
    np_count = 0
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_normalized[y_val][x_val]
        if c > 0.01:
            if (x_val == y_val):
                plt.text(x_val, y_val, "%0.2f" % (c * 100,), color='white', fontsize=12, \
                         va='center', ha='center')
                np1[np_count] = c*100
                np_count += 1
            else:
                plt.text(x_val, y_val, "%0.2f" % (c * 100,), color='black', fontsize=12, \
                         va='center', ha='center')
        plt.xticks(x_location, np.arange(classes))
    print(np1)
    df = pd.DataFrame(np1)
    df = df.T
    df.to_excel('C:\\Users\\user\\Desktop\\excel_output.xls')

    plt.yticks(x_location, np.arange(classes))
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')
    plt.tick_params(axis=u'both', which=u'both', length=0)
    plt.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
    plt.colorbar()
    plt.show()
    if sasve_file:
        plt.savefig()

def new_read_dataset(data_path, sense, gesture, data_type, times, frame_num):
    data = []
    label = []
    print('Sense:', sense)
    print('==============================')
    for g in range(gesture):
        filename = data_type + '_S' + str(sense) + 'G' + str(g) + '.npy'
        data_tmp = np.load(data_path + filename)
        data_len = (np.shape(data_tmp)[0])
        # print("data_type is  :"+str(np.shape(data_tmp)))
        print(str(data_len))
        label_tmp = np.zeros(data_len*64)
        label_tmp = label_tmp + g
        data.extend(data_tmp)
        label.extend(label_tmp)
    print('==============================')

    if data_type == 'RAI':
        data = np.reshape(data, [np.shape(data)[0], np.shape(data)[1], np.shape(data)[2], -1])
        # data = np.reshape(data, [np.shape(data)[0], np.shape(data)[1], -1])
    #     data = data[0:720,:,:,:,:]

    # onehotencoder = OneHotEncoder(categorical_features=[0])
    # data_str_ohe = onehotencoder.fit_transform(data).toarray()
    # label = data_str_ohe
    # label = np.reshape(label, (np.shape(data)[0], 64))
    # label = to_categorical(label)
    data = np.expand_dims(data, axis=-1)
    print("data length is :"+str(np.shape(data)))
    print("label length is :"+str(np.shape(label)))
    # label = np.array(label)
    label = np.reshape(label, [np.shape(data)[0], 64])
    label = to_categorical(label)
    return data, label
