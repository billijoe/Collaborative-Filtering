import sys
from PyQt5 import QtWidgets  # 从PyQt库导入QtWidget通用窗口类


class QtTestWindow(QtWidgets.QWidget):
    # QtTestWindow类继承QtWidgets.QWidget类
    def __init__(self):
        # 重载类初始化函数
        super(QtTestWindow, self).__init__()  # super关键字自行百度


# pyqt窗口必须在QApplication方法中使用
app = QtWidgets.QApplication(sys.argv)

myWin = QtTestWindow()  # 创建自定义的窗体类对象

myWin.show()  # 调用窗口显示

sys.exit(app.exec_())  # 启动事件循环