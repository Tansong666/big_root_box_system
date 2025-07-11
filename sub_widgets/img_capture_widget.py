from PyQt5.QtWidgets import QWidget, QGraphicsScene, QApplication, QFileDialog
from PyQt5.QtCore import Qt, QObject, pyqtSignal, QThreadPool
from PyQt5.QtGui import QPixmap, QImage, QTextCursor
import sys
import time
import serial as ser
import numpy as np
import psutil
# # 将项目根目录添加到 sys.path
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.insert(0, project_root)

from ui.Ui_img_capature_widget import Ui_ImgCaptureWidget
from threads.img_capture_thread import ImgCapture_Thread
from signals.global_signals import signals
from signals import global_vars

# # 声明全局变量
# com_arduino = None
# com_scanning = None
# direction = 0

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
        self.init_data() # 用于初始化控件数据
        ## 3.初始化线程
        self.img_capture_thread = ImgCapture_Thread(self.savepath,self.startcode)  ### 扫描二维码同时创建目录,移动，开关灯，图像拍照，图像保存，发送信号
        # self.thread_manager = QThreadPool()  # 用于管理线程池
        ## 4.初始化事件
        self.init_ui()
        ## 5.初始化重定向
        sys.stdout = EmittingStr(textWritten=self.outputWritten) # 用于将输出重定向到textBrowser中
        sys.stdin = EmittingStr(textWritten=self.outputWritten)  
    
    def init_data(self): # 用于初始化控件数据
        # 初始化重拍开始二维码
        self.ui.cb_missCode.addItem("None")
        self.ui.cb_missCode.addItem("451")
        self.ui.cb_missCode.setCurrentIndex(0)
        self.startcode = self.ui.cb_missCode.currentText() # 用于设置开始扫描位置
        # 初始化二维码位置进度条
        self.ui.spin_nowCode.setValue(0) # 用于设置当前扫描位置
        self.ui.spin_sumCode.setValue(810) # 用于设置总扫描位置
        self.ui.spin_nowCode.setMaximum(self.ui.spin_sumCode.value())
        self.ui.prg_grab.setValue(self.ui.spin_nowCode.value()) # 用于设置进度条的值
        self.ui.prg_grab.setMaximum(self.ui.spin_sumCode.value()) # 用于设置进度条的最大值
        # 初始化图像保存路径
        t = time.gmtime()
        t_md = time.strftime("%m%d", t)
        self.savepath = "/home/ts/Documents/RootData/Date" + t_md + "_right"
        self.ui.txt_saveDir.setText(self.savepath) # 用于设置保存路径
        # 初始化是否自动处理图像
        self.is_process = self.ui.chk_isProcessImg.isChecked()
        # 初始化场景视图
        # self.scene = QGraphicsScene()
        # self.ui.view_grabImg1.setScene(self.scene)
        # self.ui.view_grabImg1.show()
        # 更新磁盘状态
        self.update_disk_status()

    def outputWritten(self, text): # 用于将输出重定向到textBrowser中
        cursor = self.ui.txtBrw_log.textCursor() # 用于获取文本光标
        cursor.movePosition(QTextCursor.End) # 用于将光标移动到文本的末尾
        cursor.insertText(text) # 用于在光标处插入文本
        self.ui.txtBrw_log.setTextCursor(cursor) # 用于设置文本光标
        self.ui.txtBrw_log.ensureCursorVisible() # 用于确保文本光标可见

    def update_textBrowser(self, text):
        # current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        # self.ui.txtBrw_log.append(current_time + ' >>' + str(text))
        self.ui.txtBrw_log.append(str(text))
        self.ui.txtBrw_log.moveCursor(self.ui.txtBrw_log.textCursor().End)
        # self.ui.txtBrw_log.repaint()
        self.ui.txtBrw_log.ensureCursorVisible()

    def update_graphicsview1(self, image_path, image): # 用于更新图形场景
        graph_scene = QGraphicsScene() # 用于创建一个图形场景对象
        self.ui.view_grabImg1.setScene(graph_scene) # 用于设置图形场景
        if isinstance(image, str): # 用于判断image的类型   用于检查一个对象是否是一个类的实例,image是不是str（类）的实例
            image = QPixmap(image) # # 创建 QPixmap 对象并加载图像
        elif isinstance(image, QPixmap):  
            pass
        elif isinstance(image, np.ndarray):  
            image = QPixmap(QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)) # 创建 QPixmap 对象并加载ndarry图像
        else:
            raise TypeError('image type not supported') # 用于抛出异常
        # 可以使用 QPixmap 对象的各种方法来操作图像，如缩放、剪裁、旋转等。例如，scaled() 方法可以用来缩放图像：
        image = image.scaled(self.ui.view_grabImg1.width() - 3, self.ui.view_grabImg1.height() - 3, 
                            Qt.KeepAspectRatio)
        graph_scene.addPixmap(image) 
        graph_scene.update()
        self.ui.lbl_saveInfo1.setText(image_path)


    def update_graphicsview2(self, image_path, image):   # 用于更新图形场景
        graph_scene = QGraphicsScene() 
        self.ui.view_grabImg2.setScene(graph_scene)
        if isinstance(image, str):
            image = QPixmap(image)
        elif isinstance(image, QPixmap):
            pass
        elif isinstance(image, np.ndarray):
            image = QPixmap(QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888))
        else:
            raise TypeError('image type not supported')
        image = image.scaled(self.ui.view_grabImg2.width() - 3, self.ui.view_grabImg2.height() - 3,
                            Qt.KeepAspectRatio)
        graph_scene.addPixmap(image)
        graph_scene.update()
        self.ui.lbl_saveInfo2.setText(image_path)

    def update_manual_sumCode(self): # 用于更新进度条
        self.ui.prg_grab.setMaximum(self.ui.spin_sumCode.value()) # 用于设置进度条的最大值
        self.ui.spin_nowCode.setMaximum(self.ui.spin_sumCode.value())

    def update_progressBar(self, value): # 用于更新进度条
        self.ui.spin_nowCode.setValue(int(value)) # 用于设置进度条的值
        self.ui.prg_grab.setValue(int(value))
    def update_manual_progressBar(self): # 用于更新进度条
        self.ui.prg_grab.setValue(self.ui.spin_nowCode.value())

    def update_savepath(self): # 用于更新保存路径
        self.savepath = self.ui.txt_saveDir.text() # 用于获取保存路径
        print('The filedir path is'+ self.savepath)

    def update_startCode(self): # 用于更新开始位置
        self.startcode = self.ui.cb_missCode.currentText()
    
    def update_isProcess(self): # 用于更新是否处理图像
        self.is_process = self.ui.chk_isProcessImg.isChecked()

    def choose_savepath(self): # 用于选择图像保存路径
        path = QFileDialog.getExistingDirectory(None, "请选择文件夹路径", ".")  # 用于选择图像保存路径
        self.savepath = path  
        print('The filedir path is ' + path)

    def convert_bytes(self, bytes_value):
        """
        将字节数转换为带单位的字符串
        """
        if bytes_value >= 1024**3:
            return f"{bytes_value / (1024**3):.2f} GB"
        elif bytes_value >= 1024**2:
            return f"{bytes_value / (1024**2):.2f} MB"
        elif bytes_value >= 1024:
            return f"{bytes_value / 1024:.2f} KB"
        else:
            return f"{bytes_value} B"

    def update_disk_status(self):
        """
        获取当前磁盘空间状态。
        :return: 包含磁盘总空间、已使用空间、可用空间和使用率的字典。
        """
        disk_usage = psutil.disk_usage('.')
        self.disk_free = disk_usage.free
        display_text = f"{self.convert_bytes(disk_usage.free)}（{self.convert_bytes(disk_usage.used)}/{self.convert_bytes(disk_usage.total)}）"
        self.ui.lbl_diskInfo.setText(display_text)

    def serial_isopen(self): # 用于打开串口
        comboBox_arduino = self.ui.cb_arduino.currentText() # 用于获取串口号
        try:
            global_vars.com_arduino = ser.Serial(comboBox_arduino, 9600, timeout=0.5) # 用于打开串口01
            if global_vars.com_arduino.isOpen():
                print('arduino串口打开成功')
            else:
                print('arduino串口打开失败')
        except:
            print('arduino串口打开失败')

        comboBox_scan = self.ui.cb_scanner.currentText()
        try:
            global_vars.com_scanning = ser.Serial(comboBox_scan, 115200, timeout=0.5)
            if global_vars.com_scanning.isOpen():
                print('二维码串口打开成功')
            else:
                print('二维码串口打开失败')
        except:
            print('二维码串口打开失败')

    def go_forward(self): # 用于向前
        try:
            global_vars.com_scanning.flushInput()  # 用于清空串口缓冲区
        except AttributeError:
            print(f"扫码器串口未打开---{AttributeError}:'NoneType' object has no attribute 'flushInput'")
            return
        if self.disk_free < 50 * 1024**3: # 用于判断磁盘空间是否小于50GB
            print('磁盘空间不足')
            # return
        
        self.img_capture_thread.signal_img1_set.connect(self.update_graphicsview1) ##### todo 自定义信号，接收emit（image）信号，传递image参数，用于更新图形场景
        self.img_capture_thread.signal_img2_set.connect(self.update_graphicsview2) # 用于更新图形场景
        # self.thread_manager.start(self.thread_capture)
        global_vars.direction = 1
        self.img_capture_thread.savepath = self.savepath
        self.img_capture_thread.startcode = self.startcode
        self.img_capture_thread.is_process = self.is_process
        print(self.startcode)
        print(self.is_process)
        self.img_capture_thread.start()

    def go_back(self): # 用于向后
        global_vars.com_scanning.flushInput()
        if self.disk_free < 50 * 1024**3: # 用于判断磁盘空间是否小于50GB
            print('磁盘空间不足')
            # return
        self.img_capture_thread.signal_img1_set.connect(self.update_graphicsview1)
        self.img_capture_thread.signal_img2_set.connect(self.update_graphicsview2)
        # self.thread_manager.start(self.thread_capture)
        global_vars.direction = 2
        self.img_capture_thread.savepath = self.savepath
        self.img_capture_thread.startcode = self.startcode
        self.img_capture_thread.is_process = self.is_process
        self.img_capture_thread.start()

    def stop(self): # 用于停止
        global_vars.direction = 3
        print('停止扫描')

    def test(self): # 用于测试
        if self.is_process is True:
            print('开始处理根系测试图像...')
            # img1_save_path = "/home/ts/Documents/RootBoxSystem/RootBoxSystem_v1/data/root_data_demo/ori_captured/0510/171/1.png"  # linux下路径
            # img2_save_path = "/home/ts/Documents/RootBoxSystem/RootBoxSystem_v1/data/root_data_demo/ori_captured/0510/171/2.png"
            img1_save_path = r"E:\big_root_system\data\171\1.png"  # windows下路径
            img2_save_path = r"E:\big_root_system\data\171\2.png"
            # img1 = cv2.imread(img1_save_path)
            # img2 = cv2.imread(img2_save_path)
            # signals.img_process_signal.emit(img1_save_path, img2_save_path, img1, img2)
            signals.img_process_path_signal.emit(img1_save_path,img2_save_path)
        else:
            print('请勾选自动处理图像选项进行图像处理测试')

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            global_vars.direction = 3
            print('停止扫描')

    def init_ui(self):
        self.ui.txt_saveDir.returnPressed.connect(self.update_savepath) #  returnPressed 和 editingFinished
        # 添加事件
        self.ui.btn_serialConnect.clicked.connect(self.serial_isopen) # 用于打开串口 
        self.ui.btn_forward.clicked.connect(self.go_forward) # 用于向前
        self.ui.btn_back.clicked.connect(self.go_back)  # 用于向后
        self.ui.btn_stop.clicked.connect(self.stop) # 用于停止
        self.ui.btn_test.clicked.connect(self.test) # 用于测试
        # self.ui.cb_mode.currentIndexChanged.connect(self.on_mode_changed)
        # signals.code_scan_signal.connect(self.update_progressBar)
        self.ui.spin_sumCode.valueChanged.connect(self.update_manual_sumCode)
        self.ui.spin_nowCode.valueChanged.connect(self.update_manual_progressBar) 
        self.ui.cb_missCode.currentIndexChanged.connect(self.update_startCode)
        self.ui.chk_isProcessImg.stateChanged.connect(self.update_isProcess)

        signals.code_signal.connect(self.update_progressBar)

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = ImgCaptureWidget()
    window.show()
    
    sys.exit(app.exec_())
