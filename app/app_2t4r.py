from PyQt5 import QtCore, QtGui, QtWidgets
from pyqtgraph import GraphicsLayoutWidget

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
        self.graphicsView_3 = GraphicsLayoutWidget(self.centralwidget)
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
        self.pushButton_exit = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_exit.setGeometry(QtCore.QRect(1170, 670, 90, 30))
        font = QtGui.QFont()
        font.setFamily("微軟正黑體 Light")
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.pushButton_exit.setFont(font)
        self.pushButton_exit.setObjectName("pushButton_exit")

        self.pushButton_start = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_start.setGeometry(QtCore.QRect(30, 670, 150, 30))
        font = QtGui.QFont()
        font.setFamily("微軟正黑體 Light")
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(100)
        self.pushButton_start.setFont(font)
        self.pushButton_start.setObjectName("pushButton_start")
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
        self.pushButton_exit.setText(_translate("MainWindow", "Exit"))
        self.pushButton_start.setText(_translate("MainWindow", "Send Radar Config"))