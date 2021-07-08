# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'D:\QT_design\QT_design.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QTextEdit
from pyqtgraph import GraphicsLayoutWidget
import pyqtgraph.opengl as gl



class Ui_MainWindow(object):

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1920, 720)
        font = QtGui.QFont()
        font.setFamily("MS Gothic")
        font.setPointSize(10)
        MainWindow.setFont(font)
        MainWindow.setContextMenuPolicy(QtCore.Qt.NoContextMenu)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.graphicsView = GraphicsLayoutWidget(self.centralwidget)
        self.graphicsView.setGeometry(QtCore.QRect(30, 60, 600, 600))
        self.graphicsView.setObjectName("graphicsView")
        self.graphicsView_2 = GraphicsLayoutWidget(self.centralwidget)
        self.graphicsView_2.setGeometry(QtCore.QRect(660, 60, 600, 600))
        self.graphicsView_2.setObjectName("graphicsView_2")
        # self.graphicsView_3 = GraphicsLayoutWidget(self.centralwidget)
        # self.graphicsView_3.setGeometry(QtCore.QRect(1290, 60, 600, 600))
        # self.graphicsView_3.setObjectName("graphicsView_3")

        self.graphicsView_3 = gl.GLViewWidget(self.centralwidget)
        self.graphicsView_3.setGeometry(QtCore.QRect(1290, 60, 600, 600))
        self.graphicsView_3.setObjectName("graphicsView_3")


        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(210, 20, 200, 30))
        font = QtGui.QFont()
        font.setFamily("微軟正黑體 Light")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")

        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(840, 20, 200, 30))
        font = QtGui.QFont()
        font.setFamily("微軟正黑體 Light")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")

        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(1470, 20, 200, 30))
        font = QtGui.QFont()
        font.setFamily("微軟正黑體 Light")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")

        self.pushButton_exit = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_exit.setGeometry(QtCore.QRect(1270, 670, 90, 30))
        font = QtGui.QFont()
        font.setFamily("微軟正黑體 Light")
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.pushButton_exit.setFont(font)
        self.pushButton_exit.setObjectName("pushButton_exit")

        self.pushButton_save = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_save.setGeometry(QtCore.QRect(1170, 670, 90, 30))
        font = QtGui.QFont()
        font.setFamily("微軟正黑體 Light")
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.pushButton_save.setFont(font)
        self.pushButton_save.setObjectName("pushButton_save")

        self.pushButton_record = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_record.setGeometry(QtCore.QRect(970, 670, 90, 30))
        font = QtGui.QFont()
        font.setFamily("微軟正黑體 Light")
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.pushButton_record.setFont(font)
        self.pushButton_record.setObjectName("pushButton_record")

        self.pushButton_stop_record = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_stop_record.setGeometry(QtCore.QRect(1070, 670, 90, 30))
        font = QtGui.QFont()
        font.setFamily("微軟正黑體 Light")
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.pushButton_stop_record.setFont(font)
        self.pushButton_stop_record.setObjectName("pushButton_stop_record")

        self.pushButton_start = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_start.setGeometry(QtCore.QRect(190, 670, 150, 30))
        font = QtGui.QFont()
        font.setFamily("微軟正黑體 Light")
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(100)
        self.pushButton_start.setFont(font)
        self.pushButton_start.setObjectName("pushButton_start")

        self.pushButton_DCA = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_DCA.setGeometry(QtCore.QRect(30, 670, 150, 30))
        font = QtGui.QFont()
        font.setFamily("微軟正黑體 Light")
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(100)
        self.pushButton_DCA.setFont(font)
        self.pushButton_DCA.setObjectName("pushButton_DCA")


        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(350, 670, 600, 30))
        font = QtGui.QFont()
        font.setFamily("微軟正黑體 Light")
        # font.setPointSize(14)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(1000)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")

        # self.label_4 = QtWidgets.QLabel(self.centralwidget)
        # self.label_4.setGeometry(QtCore.QRect(350, 670, 120, 30))
        # font = QtGui.QFont()
        # font.setFamily("微軟正黑體 Light")
        # # font.setPointSize(14)
        # font.setBold(True)
        # font.setItalic(False)
        # font.setWeight(1000)
        # self.label_4.setFont(font)
        # self.label_4.setObjectName("label_4")

        # self.textEdit = QTextEdit(self.centralwidget)
        # self.textEdit.setGeometry(QtCore.QRect(780, 670, 150, 30))
        # self.textEdit.setFont(font)
        # self.textEdit.setObjectName("textEdit_save")


        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Real-Time Radar"))
        self.label.setText(_translate("MainWindow", "Range-Doppler Image"))
        self.label_2.setText(_translate("MainWindow", "Range-Angle Image"))
        self.label_3.setText(_translate("MainWindow", "Save File Path & Name:"))
        self.label_6.setText(_translate("MainWindow", "Point Cloud"))
        self.pushButton_exit.setText(_translate("MainWindow", "Exit"))
        self.pushButton_record.setText(_translate("MainWindow", "Record"))
        self.pushButton_start.setText(_translate("MainWindow", "Send Radar Config"))
        self.pushButton_stop_record.setText(_translate("MainWindow", "Stop Record"))
        self.pushButton_save.setText(_translate("MainWindow", "Save"))
        self.pushButton_DCA.setText(_translate("MainWindow", "Connect DCA1000"))

