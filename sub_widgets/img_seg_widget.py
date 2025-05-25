# 系统模块
import signal
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import sys
import os
import time
import cv2
# import fastdeploy as fd
import numpy as np


# 将项目根目录添加到 sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
# 自定义模块
# from common import utils
from ui.Ui_img_seg_widget import Ui_ImgSegWidget
# from threads.img_seg_thread import ImgSegThread
from threads.img_seg_thread_v2 import ImgSegThread
from threads.model_load_thread import ModelLoadThread
# from mygraphicsview import MyGraphicsView
# from threads.img_capture_thread import ImgCapture_Thread
from signals.global_signals import signals  # 导入全局信号实例

# 重定向类
class MessagePrinter(QObject):
    text_written = pyqtSignal(str)
    def write(self, text):
        self.text_written.emit(text)
    def flush(self):
        self.text_written.emit('')  # 清空缓冲区时发送空字符串，以确保清空文本框中的内容

class ImgSegWidget(QWidget):
    def __init__(self, parent=None):
        # 1. 初始化父类和ui
        super().__init__(parent)
        # self.main_window: QMainWindow = parent
        self.ui = Ui_ImgSegWidget()
        self.ui.setupUi(self)

        # 2.初始化控件数据
        self.init_data()
        # 3.初始化分割线程
        self.load_model_thread = ModelLoadThread(model_path=self.ui.lineEdit_weight_dir.text())
        self.img_seg_thread = ImgSegThread()  # 初始化线程对象
        # self.img_seg_thread.signal_process_complete.connect(self.update_textBrowser)  # 连接信号和槽函数，用于更新日志窗口
        # self.ui.graphicsView_Left.setScene(QGraphicsScene())  # 初始化场景
        # self.ui.graphicsView_Right.setScene(QGraphicsScene())  # 初始化场景
        # 连接信号，用于更新视图
        self.img_seg_thread.signal_concat_complete.connect(self.update_graphicsView_Left)
        self.img_seg_thread.signal_seg_complete.connect(self.update_graphicsView_Right)
        self.img_seg_thread.signal_process_complete.connect(self.update_textBrowser)  # 连接信号和槽函数，用于更新日志窗口      
        # 4.初始化事件
        self.init_ui()

        # 4.将输出信息重定向到日志窗口
        # self.message_printer = MessagePrinter()
        # self.message_printer.text_written.connect(self.append_text_to_browser)
        # sys.stdout = self.message_printer
        # sys.stderr = self.message_printer
 
    def update_graphicsView_Left(self, img_path, concat_matrix):
        """更新 graphicsView_Left 显示拼接后的图像"""
        concat_matrix = cv2.rotate(concat_matrix, cv2.ROTATE_90_COUNTERCLOCKWISE)  # 逆时针旋转90度
        height, width, channel = concat_matrix.shape
        bytes_per_line = 3 * width
        q_img = QImage(concat_matrix.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.ui.graphicsView_Left.setPixmap(QPixmap.fromImage(q_img))

        self.ui.label_imgPath1.setText(img_path)
        self.ui.label_ImageSize.setText(f"image size:{width}x{height}")

    def update_graphicsView_Right(self, img_path, seg_mask):
        """更新 graphicsView_Right 显示分割后的图像"""
        seg_mask = cv2.rotate(seg_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)  # 逆时针旋转90度
        height, width = seg_mask.shape
        q_img = QImage(seg_mask.data, width, height, width, QImage.Format_Grayscale8)
        self.ui.graphicsView_Right.setPixmap(QPixmap.fromImage(q_img))

        self.ui.label_imgPath2.setText(img_path)
    ## 重定向方法
    # 1.根据需要重定向输出信息到日志窗口
    def update_textBrowser(self, text):
        if self.ui.splitter_Right.sizes()[1] == 0:
            self.ui.splitter_Right.setSizes([1, 1])
        if self.ui.splitter_Middle.sizes()[1] == 0:
            self.ui.splitter_Middle.setSizes([1, 1])
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        self.ui.textBrowser_MessagePrinter.append(current_time + ' >>' + str(text))
        self.ui.textBrowser_MessagePrinter.moveCursor(self.ui.textBrowser_MessagePrinter.textCursor().End)
        self.ui.textBrowser_MessagePrinter.repaint()
    # 2.根据终端输出信息更新日志窗口
    def append_text_to_browser(self, text):
        self.ui.textBrowser_MessagePrinter.append(text)
    def clear_logs(self):
        """清空日志窗口的内容"""
        self.ui.textBrowser_MessagePrinter.clear()
    

    def load_treeWidget_from_dirpath(self, dir_path, treeWidget):
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            if os.path.isdir(file_path):
                dir_item = QTreeWidgetItem(treeWidget)
                dir_item.setText(0, file_name)
                dir_item.setText(1, file_path)
                self.load_treeWidget_from_dirpath(file_path, dir_item)
                if dir_item.childCount() == 0:
                    if type(treeWidget) == QTreeWidget:
                        treeWidget.takeTopLevelItem(treeWidget.indexOfTopLevelItem(dir_item))
                    else:
                        treeWidget.removeChild(dir_item)
            elif os.path.isfile(file_path) and file_path.endswith(image_extensions):
                # self.file_list.append(file_path)
                self.file_dict[file_path] = {'binary_path': None, 'processed_path': None, 'traits': None,
                                             'visualization': None}
                item = QTreeWidgetItem(treeWidget)
                item.setText(0, os.path.basename(file_path))
                item.setText(1, file_path)

    def select_dir_path(self):
        dir_path = QFileDialog.getExistingDirectory(self, 'Open dir')
        if dir_path == '':
            return
        sender = self.sender()  # 返回发送当前信号的对象（控件）
        if sender == self.ui.btn_SaveDir_concat:
            self.ui.lineEdit_SaveDir_concat.setText(dir_path)
        elif sender == self.ui.lineEdit_SaveDir_seg.setText(dir_path):
            self.ui.lineEdit_SaveDir_seg.setText(dir_path)
        elif sender == self.ui.pushButton_DataDir:
            self.dataset_rootpath = dir_path
            self.ui.label_DataDirShow.setText(dir_path)
            # self.file_list = []
            self.file_dict = {}
            self.ui.treeWidget_Files.clear()
            self.load_treeWidget_from_dirpath(dir_path, self.ui.treeWidget_Files)
            self.ui.treeWidget_Files.expandAll()  # 展开所有节点
            self.current_image = {'image_path': None, 'img': None, 'thresholdseg': None, 'gray': None,
                                  'processed': None, 'binary': None, 'traits': None, 'visualization': None}
            # self.update_file_dict(self.lineEdit_SaveDir_predict)
            # self.update_file_dict(self.lineEdit_SaveDir_postporcess)
            # self.update_file_dict(self.lineEdit_SaveDir_calculate)
            # self.update_current_image()
            # self.inpainting_mode('quit')
        # elif sender == self.pushButton_SaveDir_postporcess:
        #     self.lineEdit_SaveDir_postporcess.setText(dir_path)
        # elif sender == self.pushButton_SaveDir_cacuate:
        #     self.lineEdit_SaveDir_calculate.setText(dir_path)
        elif sender == self.ui.btn_weight_dir:
            self.ui.lineEdit_weight_dir.setText(dir_path)
    
    def select_file_path(self):
        file_path = QFileDialog.getOpenFileName(self, 'Open file', './')
        if file_path[0] == '':
            return
        sender = self.sender()
        if sender == self.ui.btn_weight_dir:
            self.ui.lineEdit_weight_dir.setText(file_path[0])

    def update_x1(self):
        self.x1 = self.ui.spinBox_x1.value()  # 获取spinBox_x1的值
        print("x1:", self.x1)  # 打印x1的值

    def update_x2(self):
        self.x2 = self.ui.spinBox_x2.value()  # 获取spinBox_x2的值
        print("x2:", self.x2)  # 打印x2的值

    def update_x3(self):
        self.x3 = self.ui.spinBox_x3.value()  # 获取spinBox_x3的值
        print("x3:", self.x3)  # 打印x3的值

    # 更新seg_mode的值
    def update_seg_mode(self):
        if self.ui.radioButton_edge.isChecked():  # 判断radioButton_edge是否被选中
            self.seg_mode = 'edge'  # 如果被选中，将seg_mode设置为'edge'
            self.ui.tabWidget_seg.setCurrentIndex(0)  # 设置tabWidget_seg的当前索引为0

        elif self.ui.radioButton_serving.isChecked():  # 判断radioButton_serving是否被选中
            self.seg_mode = 'serving'  # 如果被选中，将seg_mode设置为'serving'
            self.ui.tabWidget_seg.setCurrentIndex(1)  # 设置tabWidget_seg的当前索引为1
        print("seg_mode:", self.seg_mode)  # 打印seg_mode的值

    # 更新runtime_mode的值
    def update_runtime_mode(self):
        self.runtime_mode = self.ui.cb_runtime_opt.currentText()  # 获取cb_runtime_opt的当前文本值
        print("runtime_mode:", self.runtime_mode)  # 打印runtime_mode的值

    # 更新seg_model的值
    def update_seg_model(self):
        self.seg_model = self.ui.cb_seg_model_opt.currentText()  # 获取cb_model_seg的当前文本值
        print("seg_model:", self.seg_model)  # 打印seg_model的值

    # 更新IsSlide的值
    def update_IsSlide(self):
        if self.ui.checkBox_IsSlide.isChecked():
            self.ui.spinBox_CropSize_predict.setEnabled(True)
            self.ui.spinBox_Stride.setEnabled(True)
        else:
            self.ui.spinBox_CropSize_predict.setEnabled(False)
            self.ui.spinBox_Stride.setEnabled(False)

    # 更新IsResize的值
    def update_IsResize(self):
        if self.ui.checkBox_IsResize.isChecked():
            self.ui.doubleSpinBox_resize.setEnabled(True)
        else:
            self.ui.doubleSpinBox_resize.setEnabled(False)

    def on_model_loaded(self, model):
        # start_time = time.time()
        self.img_seg_thread.model = model
        # end_time = time.time()
        # self.update_textBrowser(f'传模型耗时: {end_time - start_time:.2f} 秒')
        self.update_textBrowser('模型加载完成')
        # self.seg_thread()

    def on_load_error(self, error):
        self.update_textBrowser(f'模型加载失败: {error}') 

    def record(self, start_time, end_time):
        self.update_textBrowser(f'模型加载耗时: {end_time - start_time:.2f} 秒')

    def load_model(self):
        self.load_model_thread.model_path = self.ui.lineEdit_weight_dir.text()
        self.load_model_thread.device = 'gpu'
        self.load_model_thread.use_trt = True
        self.load_model_thread.use_paddle_trt = False
        # self.load_model_thread = ModelLoadThread(model_path, device, use_trt, use_paddle_trt)
        self.load_model_thread.model_loaded.connect(self.on_model_loaded)
        self.load_model_thread.load_error.connect(self.on_load_error)
        # self.load_model_thread.record.connect(self.record)  # 连接信号和槽函数，用于更新日志窗口
        self.load_model_thread.start()


    def seg_thread(self):
        if self.img_seg_thread.model is None:  # 判断模型是否加载完成
            self.update_textBrowser('请先加载模型')
            return
        # from threads.img_seg_thread import ImgSegThread
        # 获取相关控件数据
        if not self.ui.radioButton_serving.isChecked():
            args = {'is_edge': True}
            args['device'] = 'gpu'
            args['use_trt'] = False
            args['use_paddle_trt'] = False

            args['img_path'] = self.ui.label_DataDirShow.text()
            args['concat_save_path'] = self.ui.lineEdit_SaveDir_concat.text()
            args['x1'] = self.ui.spinBox_x1.value()
            args['x2'] = self.ui.spinBox_x2.value()
            args['x3'] = self.ui.spinBox_x3.value()

            args['model_path'] = self.ui.lineEdit_weight_dir.text()
            args['slide_predict'] = self.ui.checkBox_IsSlide.isChecked()
            args['slide_size'] = [self.ui.spinBox_CropSize_predict.value(), self.ui.spinBox_CropSize_predict.value()]
            args['stride'] = [self.ui.spinBox_Stride.value(), self.ui.spinBox_Stride.value()]
            args['resize_predict'] = self.ui.checkBox_IsResize.isChecked()  # 这里需要根据实际情况设置
            args['resize_scale'] = self.ui.doubleSpinBox_resize.value()  # 这里需要根据实际情况设置
            args['seg_save_path'] = self.ui.lineEdit_SaveDir_seg.text()
        else:
            args = {'is_edge': False}

        # 图像拼接+分割线程
        # self.seg_thread = ImgSegThread()
        # 打印所有参数
        self.img_seg_thread.args = args
        # self.update_textBrowser('参数如下:\n' + str(self.img_seg_thread.args))
        structured_args = "参数如下:\n"
        for key, value in args.items():
            structured_args += f"  {key}: {value}\n"
        self.update_textBrowser(structured_args)

        # self.img_seg_thread.signal_process_complete.connect(self.append_text_to_browser)
        self.img_seg_thread.start() 



    def stop_thread(self):
        if self.img_seg_thread.isRunning():
            self.img_seg_thread.stop()
            self.update_textBrowser('线程已停止')
        else:
            self.update_textBrowser('线程未运行，无需停止')

    def auto_seg_thread(self,image_path1,image_path2,img1,img2):
        if self.img_seg_thread.model is None:  # 判断模型是否加载完成
            self.update_textBrowser('请先加载模型')
            return
        # from threads.img_seg_thread import ImgSegThread
        args = {'auto_seg': True}
        # args['img1'] = img1
        # args['img2'] = img2
        # args['code'] = code
        # 获取相关控件数据
        if not self.ui.radioButton_serving.isChecked():
            args['is_edge']= True
            args['device'] = 'gpu'
            args['use_trt'] = False
            args['use_paddle_trt'] = False

            args['img_path'] = self.ui.label_DataDirShow.text()
            args['concat_save_path'] = self.ui.lineEdit_SaveDir_concat.text()
            args['x1'] = self.ui.spinBox_x1.value()
            args['x2'] = self.ui.spinBox_x2.value()
            args['x3'] = self.ui.spinBox_x3.value()

            args['model_path'] = self.ui.lineEdit_weight_dir.text()
            args['slide_predict'] = self.ui.checkBox_IsSlide.isChecked()
            args['slide_size'] = [self.ui.spinBox_CropSize_predict.value(), self.ui.spinBox_CropSize_predict.value()]
            args['stride'] = [self.ui.spinBox_Stride.value(), self.ui.spinBox_Stride.value()]
            args['resize_predict'] = self.ui.checkBox_IsResize.isChecked()  # 这里需要根据实际情况设置
            args['resize_scale'] = self.ui.doubleSpinBox_resize.value()  # 这里需要根据实际情况设置
            args['seg_save_path'] = self.ui.lineEdit_SaveDir_seg.text()
        else:
            args['is_edge'] = False

        # 图像拼接+分割线程
        # self.seg_thread = ImgSegThread()
        # 打印所有参数
        self.img_seg_thread.img1 = img1
        self.img_seg_thread.img2 = img2
        self.img_seg_thread.img1_path = image_path1
        self.img_seg_thread.img2_path = image_path2
        self.img_seg_thread.args = args
        # self.update_textBrowser('参数如下:\n' + str(self.img_seg_thread.args))
        structured_args = "参数如下:\n"
        for key, value in args.items():
            structured_args += f"  {key}: {value}\n"
        self.update_textBrowser(structured_args)

        # self.img_seg_thread.signal_process_complete.connect(self.append_text_to_browser)
        self.img_seg_thread.start()

    def init_data(self):
        from config import default_cfg as cfg
        # self.cfg = cfg
        # self.seg_model_option = cfg.seg_model_list
        self.ui.label_DataDirShow.setText(os.path.join(cfg.root_path, cfg.data_path))
        self.ui.spinBox_x1.setValue(cfg.concat_x1)
        self.ui.spinBox_x2.setValue(cfg.concat_x2)
        self.ui.spinBox_x3.setValue(cfg.concat_x3)
        self.ui.lineEdit_SaveDir_concat.setText(os.path.join(cfg.root_path, cfg.concat_savepath))

        # self.ui.radioButton_edge.setChecked(cfg.is_Edge)
        self.ui.radioButton_serving.setChecked(cfg.is_Serving)
        # edge_segment

        self.ui.cb_runtime_opt.addItems(cfg.seg_runtime_list)
        self.ui.cb_runtime_opt.setCurrentIndex(0)
        self.ui.cb_seg_model_opt.addItems(cfg.seg_model_list)  # addItems和addItem的区别
        self.ui.cb_seg_model_opt.setCurrentIndex(0)  # 注释这行代码，默认选择第一个选项
        self.ui.lineEdit_weight_dir.setText(os.path.join(cfg.root_path, cfg.seg_weightpath))
        self.ui.lineEdit_SaveDir_seg.setText(os.path.join(cfg.root_path, cfg.seg_savepath))
        self.ui.checkBox_IsSlide.setChecked(cfg.is_slide)
        self.ui.spinBox_CropSize_predict.setValue(cfg.crop_size)
        self.ui.spinBox_Stride.setValue(cfg.stride)
        self.ui.checkBox_IsResize.setChecked(cfg.is_resize)
        self.ui.doubleSpinBox_resize.setValue(cfg.resize_scale)

        # serving_segment

    def init_ui(self):
        # 获取相关控件数据
        self.ui.btn_ClearLog.clicked.connect(self.clear_logs)
        ## 添加更新事件
        # 图像拼接
        self.ui.pushButton_DataDir.clicked.connect(self.select_dir_path)
        self.ui.spinBox_x1.valueChanged.connect(self.update_x1)  # 连接信号和槽函数，当x1值改变时，调用update_x1函数
        self.ui.spinBox_x2.valueChanged.connect(self.update_x2)  # 连接信号和槽函数，当x2值改变时，调用update_x2函数
        self.ui.spinBox_x3.valueChanged.connect(self.update_x3)  # 连接信号和槽函数，当x3值改变时，调用update_x3函数
        self.ui.btn_SaveDir_concat.clicked.connect(self.select_dir_path)
        # 图像分割
        self.ui.btn_weight_dir.clicked.connect(self.select_file_path)
        self.ui.btn_SaveDir_seg.clicked.connect(self.select_dir_path)
        self.ui.radioButton_edge.clicked.connect(self.update_seg_mode) 
        self.ui.radioButton_serving.clicked.connect(self.update_seg_mode)
        self.ui.cb_runtime_opt.currentIndexChanged.connect(self.update_runtime_mode)
        self.ui.cb_seg_model_opt.currentIndexChanged.connect(self.update_seg_model)
        self.ui.checkBox_IsSlide.stateChanged.connect(self.update_IsSlide)
        self.ui.checkBox_IsResize.stateChanged.connect(self.update_IsResize)
        ## 添加线程事件
        self.ui.btn_load_model.clicked.connect(self.load_model)  # 先加载模型
        self.ui.btn_Seg.clicked.connect(self.seg_thread)
        self.ui.btn_Stop.clicked.connect(self.stop_thread)
        # 根据定义的信号连接槽函数
        signals.img_seg_signal.connect(self.auto_seg_thread)






if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = ImgSegWidget()
    window.show()
    
    sys.exit(app.exec_())