from PyQt5.QtWidgets import QWidget, QGraphicsScene, QApplication, QFileDialog
from PyQt5.QtCore import Qt, QObject, pyqtSignal, QThreadPool
from PyQt5.QtGui import QPixmap, QImage, QTextCursor
import sys
import time
import serial as ser
import numpy as np
import os
import cv2
# 将项目根目录添加到 sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from ui.Ui_img_capature_widget import Ui_ImgCaptureWidget
from threads.img_capture_thread import ImgCapture_Thread
from signals.global_signals import signals


# 声明全局变量
com_arduino = None
com_scanning = None
direction = 0

class EmittingStr(QObject):
    textWritten = pyqtSignal(str)  # 定义一个发送str的信号
    def write(self, text): # 用于将输出重定向到textBrowser中
        self.textWritten.emit(str(text))

class ImgCaptureWidget(QWidget):
    def __init__(self, parent=None):
        ## 1. 初始化父类和ui
        super().__init__(parent)
        # self.main_window: QMainWindow = parent
        self.ui = Ui_ImgCaptureWidget()
        self.ui.setupUi(self)
        ## 2.初始化数据
        self.scene = QGraphicsScene() # 用于创建一个图形场景对象

        t = time.gmtime()
        t_md = time.strftime("%m%d", t)
        self.savepath = "/home/ts/Root_data/Data" + t_md  # todo

        self.img_capture_thread = ImgCapture_Thread(self.savepath)   ### 扫描二维码同时创建目录,移动，开关灯，图像拍照，图像保存，发送信号
        self.ui.graphicsView_img1.setScene(self.scene) # 用于设置图形场景
        self.ui.graphicsView_img1.show() # 用于显示图形场景

        self.thread_manager = QThreadPool()  # 用于管理线程池

        sys.stdout = EmittingStr(textWritten=self.outputWritten) # 用于将输出重定向到textBrowser中
        sys.stdin = EmittingStr(textWritten=self.outputWritten)  


        ## 3.初始化事件
        self.init_ui()

    def outputWritten(self, text): # 用于将输出重定向到textBrowser中
        cursor = self.ui.tb_log.textCursor() # 用于获取文本光标
        cursor.movePosition(QTextCursor.End) # 用于将光标移动到文本的末尾
        cursor.insertText(text) # 用于在光标处插入文本
        self.ui.tb_log.setTextCursor(cursor) # 用于设置文本光标
        self.ui.tb_log.ensureCursorVisible() # 用于确保文本光标可见

    def update_graphicsview1(self, image): # 用于更新图形场景
        graph_scene = QGraphicsScene() # 用于创建一个图形场景对象
        self.ui.graphicsView_img1.setScene(graph_scene) # 用于设置图形场景
        if isinstance(image, str): # 用于判断image的类型   用于检查一个对象是否是一个类的实例,image是不是str（类）的实例
            image = QPixmap(image) # # 创建 QPixmap 对象并加载图像
        elif isinstance(image, QPixmap):  
            pass
        elif isinstance(image, np.ndarray):  
            image = QPixmap(QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)) # 创建 QPixmap 对象并加载ndarry图像
        else:
            raise TypeError('image type not supported') # 用于抛出异常
        # 可以使用 QPixmap 对象的各种方法来操作图像，如缩放、剪裁、旋转等。例如，scaled() 方法可以用来缩放图像：
        image = image.scaled(self.ui.graphicsView_img1.width() - 3, self.ui.graphicsView_img1.height() - 3, 
                            Qt.KeepAspectRatio) # 用于缩放图像
        graph_scene.addPixmap(image) # 用于将图像添加到图形场景中
        graph_scene.update() # 用于更新图形场景

    def update_graphicsview2(self, image):   # 用于更新图形场景
        graph_scene = QGraphicsScene()   # 用于创建一个图形场景对象
        self.ui.graphicsView_img2.setScene(graph_scene) # 用于设置图形场景
        if isinstance(image, str):
            image = QPixmap(image)
        elif isinstance(image, QPixmap):
            pass
        elif isinstance(image, np.ndarray):
            image = QPixmap(QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888))
        else:
            raise TypeError('image type not supported')
        image = image.scaled(self.ui.graphicsView_img2.width() - 3, self.ui.graphicsView_img2.height() - 3,
                            Qt.KeepAspectRatio)
        graph_scene.addPixmap(image)
        graph_scene.update()

    def serial_isopen(self): # 用于打开串口
        global com_arduino 
        global com_scanning  
        comboBox_arduino = self.ui.cb_arduino.currentText() # 用于获取串口号
        try:
            com_arduino = ser.Serial(comboBox_arduino, 9600, timeout=0.5) # 用于打开串口01
            if com_arduino.isOpen():
                print('arduino串口打开成功')
            else:
                print('arduino串口打开失败')
        except:
            print('arduino串口打开失败')

        comboBox_scan = self.ui.cb_scanner.currentText()
        try:
            com_scanning = ser.Serial(comboBox_scan, 115200, timeout=0.5)
            if com_scanning.isOpen():
                print('二维码串口打开成功')
            else:
                print('二维码串口打开失败')
        except:
            print('二维码串口打开失败')

    def choose_savepath(self): # 用于选择图像保存路径
        path = QFileDialog.getExistingDirectory(None, "请选择文件夹路径", ".")  # 用于选择图像保存路径
        self.savepath = path  
        print('The filedir path is ' + path) 

    def go_forward(self): # 用于向前
        global direction
        global com_scanning
        com_scanning.flushInput()   # 用于清空串口缓冲区
        self.img_capture_thread.signal_img1_set.connect(self.update_graphicsview1) ##### todo 自定义信号，接收emit（image）信号，传递image参数，用于更新图形场景
        self.img_capture_thread.signal_img2_set.connect(self.update_graphicsview2) # 用于更新图形场景
        # self.thread_manager.start(self.thread_capture)
        direction = 1
        self.img_capture_thread.start()

    def go_back(self): # 用于向后
        global direction
        global com_scanning
        com_scanning.flushInput()
        self.img_capture_thread.signal_img1_set.connect(self.update_graphicsview1)
        self.img_capture_thread.signal_img2_set.connect(self.update_graphicsview2)
        # self.thread_manager.start(self.thread_capture)
        direction = 2
        self.img_capture_thread.start()

    def stop(self): # 用于停止
        global direction
        # self.thread_manager.start(self.thread_capture)
        direction = 3
        print('停止扫描')
        img1 = cv2.imread("E:\\big_root_system\\data\\0102\\512\\1.png")
        img2 = cv2.imread("E:\\big_root_system\\data\\0102\\512\\2.png")
        signals.signal_auto_seg.emit(img1, img2, str(512))  # emit 方法不支持关键字参数 

    def keyPressEvent(self, event):
        global direction
        if event.key() == Qt.Key_Escape:
            direction = 3
            print('停止扫描')

    def init_ui(self):
        # 获取相关控件数据
        # 添加事件
        self.ui.btn_serial_connect.clicked.connect(self.serial_isopen) # 用于打开串口 
        self.ui.btn_forward.clicked.connect(self.go_forward) # 用于向前
        self.ui.btn_back.clicked.connect(self.go_back)  # 用于向后
        self.ui.btn_stop.clicked.connect(self.stop) # 用于停止
        # self.ui.cb_mode.currentIndexChanged.connect(self.on_mode_changed)

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = ImgCaptureWidget()
    window.show()
    
    sys.exit(app.exec_())
