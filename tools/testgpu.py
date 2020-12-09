import  numpy as np
from remove_studio_header import remove_header
import sys
import pandas as pd

np1 =range(0, 12)
np_count = 0
df = pd.DataFrame(np1)
df = df.T
print(df)
df.to_excel('C:\\Users\\user\\Desktop\\excel_output.xls')
#+===================================================================
# data_folder = 'C:\\Users\\user\\Desktop\\new\\'
# data_folder1 = 'D:\\Matt_Yen\\3t4r\\'
# # file_name = data_folder1 + 'adc_data_3t4r_0_0_0_001_process_0.bin'
# file_name2 = data_folder + 'adc_data11_Raw_0.bin'
# # data = np.fromfile(file_name, dtype=np.int16)
# data2 = np.fromfile(file_name2, dtype=np.int16)
# # print(np.shape(data))
# # data = np.reshape(data, [-1, 8])
# data2 = remove_header(file_name2,4322)
# print(np.shape(data2))
# f = open(data_folder+'file11.bin', 'w+b')
# f.write(data2)
# f.close()
# sys.exit(0)

#+===================================================================#
# import numpy as np
# import matplotlib.pyplot as plt
#
# from sklearn import svm, datasets
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import plot_confusion_matrix
#
# # import some data to play with
# iris = datasets.load_iris()
# X = iris.data
# y = iris.target
# class_names = iris.target_names
#
# # Split the data into a training set and a test set
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
#
# # Run classifier, using a model that is too regularized (C too low) to see
# # the impact on the results
# classifier = svm.SVC(kernel='linear', C=0.01).fit(X_train, y_train)
#
# np.set_printoptions(precision=2)
#
# # Plot non-normalized confusion matrix
# titles_options = [("Confusion matrix, without normalization", None),
#                   ("Normalized confusion matrix", 'true')]
# for title, normalize in titles_options:
#     disp = plot_confusion_matrix(classifier, X_test, y_test,
#                                  display_labels=class_names,
#                                  cmap=plt.cm.Blues,
#                                  normalize=normalize)
#     disp.ax_.set_title(title)
#
#     print(title)
#     print(disp.confusion_matrix)
#
# plt.show()