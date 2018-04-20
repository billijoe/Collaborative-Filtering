# -*-coding:UTF-8-*-
'''
Created on 2016-8-25

@author: hongguang.jin
'''
import sys

from PyQt5 import QtCore, QtGui, Qt
from PyQt5.Qt import *
import os, sys
from PyQt5.QtGui import QFontDialog, QApplication
from toaiff import toaiff
from encodings.idna import ToASCII

QTextCodec.setCodecForTr(QTextCodec.codecForName("utf8"))

comms = {}


class StandardDialog(QDialog):
    def __init__(self, parent=None):
        super(StandardDialog, self).__init__(parent)
        self.setWindowTitle("Standard Dialog")
        reload(sys)
        sys.setdefaultencoding('utf8')

        uiFileButton = QPushButton(self.tr("选择.ui文件"))
        self.uiFileLineEdit = QLineEdit()
        uiToPyPushButton = QPushButton(self.tr("ui转为py"))

        layout = QGridLayout()
        layout.addWidget(uiFileButton, 0, 0)
        layout.addWidget(self.uiFileLineEdit, 0, 1)
        layout.addWidget(uiToPyPushButton, 0, 2)

        self.setLayout(layout)
        self.connect(uiFileButton, SIGNAL("clicked()"), self.openFile)
        self.connect(uiToPyPushButton, SIGNAL("clicked()"), self.transFile)

    def openFile(self):
        s = QFileDialog.getOpenFileName(self, "Open file dialog", "/", "Files(*.ui)")
        path = str(s).encode('utf8')
        self.uiFileLineEdit.setText(path.decode('utf8'))
        comms['name'] = path.decode('utf8').split("/")[-1].split(".")[0]
        comms['cd'] = "cd " + path.decode('utf8').split(comms['name'] + '.ui')[0]
        comms['root'] = path.decode('utf8').split("/")[0]
        comms['command'] = "Qpyuic4 " + comms['name'] + str(".ui > ") + comms['name'] + ".py"

    def transFile(self):
        f = open("trans.bat", 'w')
        f.write("@echo on\n")
        f.writelines(str(comms['root'] + "\n").encode('gbk'))
        f.writelines(str(comms['cd'] + "\n").encode('gbk'))
        f.writelines(str(comms['command'] + "\n").encode('gbk'))
        f.write("exit")
        abstractpath = os.getcwd() + '\\trans.bat'
        os.popen('start %s' % str(abstractpath))


app = QApplication(sys.argv)
form = StandardDialog()
form.show()
app.exec_()