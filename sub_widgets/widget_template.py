# 系统模块
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import sys
import os
import time
import cv2

# 将项目根目录添加到 sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 自定义模块
# from common import utils
from ui.Ui_img_postprocess_widget import Ui_PostprocessWidget
from threads.img_postprocess_thread import PostProcessThread


class ImgPostprocessWidget(QWidget):
    def __init__(self, parent=None):
        # 1. 初始化父类和ui
        super().__init__(parent)
        # self.main_window: QMainWindow = parent
        self.ui = Ui_PostprocessWidget()
        self.ui.setupUi(self)

        # 2.初始化线程
        # 大模型初始化加载模型线程

        # 3.初始化控件数据
        self.init_data()

        # 4.初始化事件
        self.init_ui()


    def update_textBrowser(self, text):
        if self.ui.splitter_post_right.sizes()[1] == 0:
            self.ui.splitter_post_right.setSizes([1, 1])
        if self.ui.splitter_post_middle.sizes()[1] == 0:
            self.ui.splitter_post_middle.setSizes([1, 1])
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        self.ui.textBrowser_MessagePrinter_post.append(current_time + ' >>' + str(text))
        self.ui.textBrowser_MessagePrinter_post.moveCursor(self.ui.textBrowser_MessagePrinter_post.textCursor().End)
        self.ui.textBrowser_MessagePrinter_post.repaint()

    def update_select_dir_path(self):
        dir_path = QFileDialog.getExistingDirectory(self, 'Open dir')
        if dir_path == '':
            return
        sender = self.sender()  # 返回发送当前信号的对象（控件）
        if sender == self.ui.btn_SaveDir_denoise:
            self.ui.lineEdit_SaveDir_denoise.setText(dir_path)
        elif sender == self.ui.btn_SaveDir_post:
            self.ui.lineEdit_SaveDir_post.setText(dir_path)
        elif sender == self.ui.btn_DataDir_post:
            self.ui.label_DataDirShow_post.setText(dir_path)
        elif sender == self.ui.btn_inpaint_weight_path:
            self.ui.lineEdit_inpaint_weight_path.setText(dir_path)

    def update_select_file_path(self):
        file_path = QFileDialog.getOpenFileName(self, 'Open file', './')
        if file_path[0] == '':
            return
        sender = self.sender()
        if sender == self.ui.btn_inpaint_weight_path:
            self.ui.lineEdit_inpaint_weight_path.setText(file_path[0])
    
    def update_processing(self):
        # 信号对象
        sender = self.sender()
        if sender == self.ui.radioButton_edge_post or sender == self.ui.radioButton_serving_post:
            if self.ui.radioButton_edge_post.isChecked():  # 判断radioButton_edge是否被选中
                self.post_mode = 'edge'  # 如果被选中，将seg_mode设置为'edge'
                self.ui.tabWidget_imgInpaint.setCurrentIndex(0)  # 设置tabWidget_seg的当前索引为0
                # self.ui.tab_deeplearning.setEnabled(False)
                self.ui.tab_serving_inpaint.setEnabled(False)
                self.ui.tab_edge_inpaint.setEnabled(True)
            elif self.ui.radioButton_serving_post.isChecked():  # 判断radioButton_serving是否被选中
                self.post_mode = 'serving'  # 如果被选中，将seg_mode设置为'serving'
                self.ui.tabWidget_imgInpaint.setCurrentIndex(1)  # 设置tabWidget_seg的当前索引为1
                # self.ui.tab_opencv.setEnabled(False)
                self.ui.tab_edge_inpaint.setEnabled(False)
                self.ui.tab_serving_inpaint.setEnabled(True)
            self.update_textBrowser('postprocess_mode:'+ self.post_mode)
            return
        # 图像去噪

        # 根系修复


    def update_graphicsView_post_Left(self, seg_mask):
        """更新 graphicsView_Right 显示分割后的图像"""
        seg_mask = cv2.rotate(seg_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)  # 逆时针旋转90度
        height, width = seg_mask.shape
        q_img = QImage(seg_mask.data, width, height, width, QImage.Format_Grayscale8)
        self.ui.post_graphicsView_Left.setPixmap(QPixmap.fromImage(q_img))
    
    def update_graphicsView_post_Right(self, denoise_mask):
        """更新 graphicsView_Left 显示拼接后的图像"""
        denoise_mask = cv2.rotate(denoise_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)  # 逆时针旋转90度
        height, width = denoise_mask.shape
        q_img = QImage(denoise_mask.data, width, height, width, QImage.Format_Grayscale8)
        self.ui.post_graphicsView_Right.setPixmap(QPixmap.fromImage(q_img))



    def start_postprocess_thread(self):
        sender = self.sender()  # 返回发送当前信号的对象（控件）
        # 从控件获取参数
        args = {}
        args['is_denoise'] = self.ui.checkBox_is_denoise.isChecked()
        if args['denoise_save_path'] == '':
            QMessageBox.warning(self, 'Warning', 'Please fill in save dir!')
            return
        inpaint_args = {}
        # inpaint_args['img_path'] = self.ui.label_DataDirShow_post.text()
        if sender == self.ui.btn_denoise:
            inpaint_args['is_inpaint'] = False
        else:
            inpaint_args['is_inpaint'] = self.ui.checkBox_is_inpaint.isChecked()
            if inpaint_args['inpaint_save_path'] == '':
                QMessageBox.warning(self, 'Warning', 'Please fill in save dir!')
                return
        # 初始化线程
        self.process_thread = PostProcessThread()
        # 定义线程信号和槽函数
        # self.ui.btn_Stop_post.clicked.connect(self.process_thread.stop)
        self.process_thread.process_signal.connect(self.update_textBrowser)
        self.process_thread.show_seg_img_signal.connect(self.update_graphicsView_post_Left)
        self.process_thread.show_post_img_signal.connect(self.update_graphicsView_post_Right)
        # 更新线程参数
        self.process_thread.args = args
        self.process_thread.inpaint_args = inpaint_args
        # 启动线程
        self.process_thread.start()

    def init_data(self):
        from config import default_cfg as cfg
        
        self.ui.radioButton_serving_post.setChecked(cfg.post_is_Serving)
        # 根系去噪

        # 根系修复
        pass

    def init_ui(self):
        ## 1.添加更新事件
        # 图像去噪
        self.ui.checkBox_is_denoise.clicked.connect(self.update_processing)
        self.ui.btn_DataDir_post.clicked.connect(self.update_select_dir_path)

        # 图像修复
        self.ui.checkBox_is_inpaint.clicked.connect(self.update_processing)
        self.ui.btn_inpaint_weight_path.clicked.connect(self.update_select_file_path)

        ## 2.添加线程事件
        # self.ui.btn_load_model.clicked.connect(self.load_model)  # 先加载模型
        self.ui.btn_denoise.clicked.connect(self.start_postprocess_thread)  # 图像去噪
        self.ui.btn_inpaint.clicked.connect(self.start_postprocess_thread)  # 图像修复


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = ImgPostprocessWidget()
    window.show()
    
    sys.exit(app.exec_())

