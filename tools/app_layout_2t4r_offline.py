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

class btn_class(QtWidgets.QPushButton):
    def __init__(self, widget, obj_name, loc_x, loc_y, w, h,group):
        super(btn_class, self).__init__(widget)
        font = QtGui.QFont()
        font.setFamily("微軟正黑體 Light")
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.setGeometry(loc_x, loc_y, w , h )
        self.setFont(font)
        self.setObjectName(obj_name)
        group.addButton(self)
#
class label_class(QtWidgets.QLabel):
    def __init__(self, widget, obj_name, loc_x, loc_y, w, h):
        super(label_class, self).__init__(widget)
        font = QtGui.QFont()
        font.setFamily("微軟正黑體 Light")
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.setGeometry(loc_x, loc_y, w, h)
        self.setFont(font)
        self.setObjectName(obj_name)

class Ui_MainWindow(object):
    def __init__(self):
        self.btn_group = QtGui.QButtonGroup()

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1920, 720)
        font = QtGui.QFont()
        font.setFamily("MS Gothic")
        font.setPointSize(10)
        MainWindow.setFont(font)
        MainWindow.setContextMenuPolicy(QtCore.Qt.NoContextMenu)
        #----------widgets setup -------------------
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.graphicsView = GraphicsLayoutWidget(self.centralwidget)
        self.graphicsView.setGeometry(QtCore.QRect(30, 60, 500, 400))
        self.graphicsView.setObjectName("graphicsView")
        self.graphicsView_2 = GraphicsLayoutWidget(self.centralwidget)
        # self.graphicsView_2 = gl.GLViewWidget(self.centralwidget)
        self.graphicsView_2.setGeometry(QtCore.QRect(560, 60, 500, 400))
        self.graphicsView_2.setObjectName("graphicsView_2")
        # self.graphicsView_3 = GraphicsLayoutWidget(self.centralwidget)
        self.graphicsView_3 = gl.GLViewWidget(self.centralwidget)
        self.graphicsView_3.setGeometry(QtCore.QRect(1090, 60, 500, 400))
        self.graphicsView_3.setObjectName("graphicsView_3")
        # ----------label setup-----------
        self.label = label_class(self.centralwidget,"label",210, 20, 200, 30)
        self.label_2 = label_class(self.centralwidget,"label_2",840, 20, 200, 30)
        self.label_3 = label_class(self.centralwidget,"label_3",30, 670, 150, 30)
        #----------btn setup-----------
        self.pd_btn = btn_class(self.centralwidget,"pd_btn",1375, 670, 90, 30,self.btn_group)
        self.browse_btn  = btn_class(self.centralwidget,"browse_btn",560,670,90,30,self.btn_group)
        self.load_btn = btn_class(self.centralwidget, "browse_btn", 660, 670, 90, 30,self.btn_group)
        self.start_btn  = btn_class(self.centralwidget,"start_btn",760,670,90,30,self.btn_group)
        self.stop_btn  = btn_class(self.centralwidget,"stop_btn",860,670,90,30,self.btn_group)
        self.next_btn  = btn_class(self.centralwidget,"next_btn",960,670,90,30,self.btn_group)
        self.pre_btn  = btn_class(self.centralwidget,"pre_btn",1060,670,90,30,self.btn_group)
        # ----------static removal checkbox setup-----------
        self.checkBox1 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox1.setChecked(True)
        self.checkBox1.setGeometry(QtCore.QRect(1160,670,200,30))
        # ----------File path textedit setup-----------
        self.textEdit = QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(150, 670, 400, 30))
        self.textEdit.setFont(font)
        self.textEdit.setObjectName("textEdit_save")

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
        self.label_3.setText(_translate("MainWindow", "Path & Name:"))
        self.pd_btn.setText(_translate("MainWindow", "Point cloud"))
        self.load_btn.setText(_translate("MainWindow", "Load file"))
        self.browse_btn.setText(_translate("MainWindow", "Browse File"))
        self.start_btn.setText(_translate("MainWindow", "Start"))
        self.stop_btn.setText(_translate("MainWindow", "Stop"))
        self.next_btn.setText(_translate("MainWindow", "Next frame"))
        self.pre_btn.setText(_translate("MainWindow", "Pre frame"))
        self.checkBox1.setText(_translate("MainWindow", "static clutter removal"))
