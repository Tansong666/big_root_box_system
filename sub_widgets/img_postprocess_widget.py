# 系统模块
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import sys
import os
import time
import cv2
import numpy as np

# 将项目根目录添加到 sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 自定义模块
# from common import utils
from ui.Ui_img_postprocess_widget import Ui_PostprocessWidget
from threads.img_postprocess_thread import PostProcessThread
from threads.phenotype_calculate_thread import PhenotypeCalculateThread
from signals.global_signals import signals


class ImgPostprocessWidget(QWidget):
    def __init__(self, parent=None):
        # 1. 初始化父类和ui
        super().__init__(parent)
        # self.main_window: QMainWindow = parent
        self.ui = Ui_PostprocessWidget()
        self.ui.setupUi(self)
        self.custom_Ui()

        # 2.初始化线程

        # 3.初始化控件数据
        self.init_data()

        # 4.初始化事件
        self.init_ui()

    def custom_Ui(self):
        # tab_calculate
        self.ui.trWItem_area = self.ui.treeWidget_Traits.topLevelItem(0)
        self.ui.trWItem_convex_area = self.ui.treeWidget_Traits.topLevelItem(1)
        self.ui.trWItem_length = self.ui.treeWidget_Traits.topLevelItem(2)
        self.ui.trWItem_diameter = self.ui.treeWidget_Traits.topLevelItem(3)
        self.ui.trWItem_depth = self.ui.treeWidget_Traits.topLevelItem(4)
        self.ui.trWItem_width = self.ui.treeWidget_Traits.topLevelItem(5)
        self.ui.trWItem_wdRatio = self.ui.treeWidget_Traits.topLevelItem(6)
        self.ui.trWItem_sturdiness = self.ui.treeWidget_Traits.topLevelItem(7)
        self.ui.trWItem_initial_x = self.ui.treeWidget_Traits.topLevelItem(8)
        self.ui.trWItem_initial_y = self.ui.treeWidget_Traits.topLevelItem(9)
        self.ui.trWItem_centroid_x = self.ui.treeWidget_Traits.topLevelItem(10)
        self.ui.trWItem_centroid_y = self.ui.treeWidget_Traits.topLevelItem(11)
        self.ui.trWItem_angle_apex_left = self.ui.treeWidget_Traits.topLevelItem(12)
        self.ui.trWItem_angle_apex_right = self.ui.treeWidget_Traits.topLevelItem(13)
        self.ui.trWItem_angle_apex_all = self.ui.treeWidget_Traits.topLevelItem(14)
        self.ui.trWItem_angle_entire_left = self.ui.treeWidget_Traits.topLevelItem(15)
        self.ui.trWItem_angle_entire_right = self.ui.treeWidget_Traits.topLevelItem(16)
        self.ui.trWItem_angle_entire_all = self.ui.treeWidget_Traits.topLevelItem(17)
        self.ui.trWItem_layer_mass = self.ui.treeWidget_Traits.topLevelItem(18)
        self.ui.trWItem_layer_mass.setExpanded(True)
        self.ui.trWItem_lmchild_Area = self.ui.trWItem_layer_mass.child(0)
        self.ui.trWItem_lmchild_Length = self.ui.trWItem_layer_mass.child(1)
        self.ui.trWItem_lmchild_Convex_hull = self.ui.trWItem_layer_mass.child(2)
        self.ui.trWItem_lmchild_A_C = self.ui.trWItem_layer_mass.child(3)
        self.ui.trWItem_lmchild_A_L = self.ui.trWItem_layer_mass.child(4)
        self.ui.trWItem_lmchild_L_C = self.ui.trWItem_layer_mass.child(5)

    def update_textBrowser(self, text):
        if self.ui.splitter_view.sizes()[1] == 0:
            self.ui.splitter_view.setSizes([1, 1])
        if self.ui.splitter_right.sizes()[1] == 0:
            self.ui.splitter_right.setSizes([1, 1])
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
        elif sender == self.ui.btn_SaveDir_caculate:
            self.ui.lineEdit_SaveDir_calculate.setText(dir_path)
        elif sender == self.ui.btn_InputDir_caculate:
            self.ui.lineEdit_InputDir_calculate.setText(dir_path)

    def update_select_file_path(self):
        file_path = QFileDialog.getOpenFileName(self, 'Open file', './')
        if file_path[0] == '':
            return
        sender = self.sender()
        if sender == self.ui.btn_inpaint_weight_path:
            self.ui.lineEdit_inpaint_weight_path.setText(file_path[0])
    
    def update_processing(self):
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
        if sender == self.ui.checkBox_is_denoise:
            if self.ui.checkBox_is_denoise.isChecked():
                self.ui.groupBox_imgDenoise.setEnabled(True)
            else:
                self.ui.groupBox_imgDenoise.setEnabled(False)
            return
        if sender == self.ui.checkBox_IsRSA:
            if self.ui.checkBox_IsRSA.isChecked():
                self.ui.spinBox_dilate_iters.setEnabled(True)
                self.ui.spinBox_threshold_area.setEnabled(True)
            else:
                self.ui.spinBox_dilate_iters.setEnabled(False)
                self.ui.spinBox_threshold_area.setEnabled(False)
            return
        if sender == self.ui.checkBox_IsRBA:
            if self.ui.checkBox_IsRBA.isChecked():
                self.ui.spinBox_rba_left.setEnabled(True)
                self.ui.spinBox_rba_right.setEnabled(True)
                self.ui.spinBox_rba_top.setEnabled(True)
                self.ui.spinBox_rba_bottom.setEnabled(True)
            else:
                self.ui.spinBox_rba_left.setEnabled(False)
                self.ui.spinBox_rba_right.setEnabled(False)
                self.ui.spinBox_rba_top.setEnabled(False)
                self.ui.spinBox_rba_bottom.setEnabled(False)
            return
        if sender == self.ui.spinBox_dilate_iters:
            self.update_textBrowser('dilate_iters: ' + str(self.ui.spinBox_dilate_iters.value()))
            return
        if sender == self.ui.spinBox_threshold_area:
            self.update_textBrowser('threshold_area:'+ str(self.ui.spinBox_threshold_area.value()))
            return
        if sender == self.ui.spinBox_rba_left:
            self.update_textBrowser('rba_left:'+ str(self.ui.spinBox_rba_left.value()))
            return
        if sender == self.ui.spinBox_rba_right:
            self.update_textBrowser('rba_right:'+ str(self.ui.spinBox_rba_right.value()))
            return
        if sender == self.ui.spinBox_rba_top:
            self.update_textBrowser('rba_top:'+ str(self.ui.spinBox_rba_top.value()))
            return
        if sender == self.ui.spinBox_rba_bottom:
            self.update_textBrowser('rba_bottom:'+ str(self.ui.spinBox_rba_bottom.value()))
            return

        # 根系修复
        if sender == self.ui.checkBox_is_inpaint:
            if self.ui.checkBox_is_inpaint.isChecked():
                self.ui.groupBox_imgInpaint.setEnabled(True)
            else:
                self.ui.groupBox_imgInpaint.setEnabled(False)
            return
        if sender == self.ui.spinBox_inpaint_iters:
            self.update_textBrowser('inpaint_iters:'+ str(self.ui.spinBox_inpaint_iters.value()))
            return

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

    def update_graphicsView_Right(self, img):
        """更新 graphicsView_Left 显示拼接后的图像"""
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)  # 逆时针旋转90度
        if len(img.shape) == 2:  # 灰度图
            height, width = img.shape
            q_img = QImage(img.data, width, height, width, QImage.Format_Grayscale8)
        elif len(img.shape) == 3:  # 彩色图
            height, width, channels = img.shape
            if channels == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                q_img = QImage(img.data, width, height, width * channels, QImage.Format_RGB888)
            elif channels == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
                q_img = QImage(img.data, width, height, width * channels, QImage.Format_RGBA8888)
            else:
                raise ValueError(f"不支持的通道数: {channels}")
        else:
            raise ValueError(f"不支持的图像形状: {img.shape}")
        
        self.ui.post_graphicsView_Right.setPixmap(QPixmap.fromImage(q_img))

    def update_traits_table(self, traits):
        # traits = self.current_image['traits']
        # if traits is None:
        #     self.traits_init()
        #     return
        self.ui.trWItem_area.setText(1, str(traits['area']))
        self.ui.trWItem_convex_area.setText(1, str(traits['convex_area']))
        self.ui.trWItem_length.setText(1, str(traits['length']))
        self.ui.trWItem_diameter.setText(1, str(traits['diameter']))
        self.ui.trWItem_depth.setText(1, str(traits['depth']))
        self.ui.trWItem_width.setText(1, str(traits['width']))
        self.ui.trWItem_wdRatio.setText(1, str(traits['wd_ratio']))
        self.ui.trWItem_sturdiness.setText(1, str(traits['sturdiness']))
        self.ui.trWItem_initial_x.setText(1, str(traits['initial_x']))
        self.ui.trWItem_initial_y.setText(1, str(traits['initial_y']))
        self.ui.trWItem_centroid_x.setText(1, str(traits["centroid_x"]))
        self.ui.trWItem_centroid_y.setText(1, str(traits['centroid_y']))
        self.ui.trWItem_angle_apex_left.setText(1, str(traits['apex_angle_left']))
        self.ui.trWItem_angle_apex_right.setText(1, str(traits['apex_angle_right']))
        self.ui.trWItem_angle_apex_all.setText(1, str(traits['apex_angle']))
        self.ui.trWItem_angle_entire_left.setText(1, str(traits['entire_angle_left']))
        self.ui.trWItem_angle_entire_right.setText(1, str(traits['entire_angle_right']))
        self.ui.trWItem_angle_entire_all.setText(1, str(traits['entire_angle']))
        self.ui.trWItem_lmchild_Area.setText(1, str(traits['layer_mass_A']))
        self.ui.trWItem_lmchild_Length.setText(1, str(traits['layer_mass_L']))
        self.ui.trWItem_lmchild_Convex_hull.setText(1, str(traits['layer_mass_C']))
        self.ui.trWItem_lmchild_A_C.setText(1, str(traits['layer_mass_A_C']))
        self.ui.trWItem_lmchild_A_L.setText(1, str(traits['layer_mass_A_L']))
        self.ui.trWItem_lmchild_L_C.setText(1, str(traits['layer_mass_L_C']))

    def calculate_one_image(self, traits):
        # self.update_progress(self.progressBar_statubar.value() + 1, self.progressBar_statubar.maximum())
        # self.current_image['image_path'] = traits['image_path']
        # self.current_image['traits'] = traits
        # self.file_dict[self.current_image['image_path']]['traits'] = traits
        # self.update_graphicsview(self.current_image['image_path'])
        # self.update_traits_table()
        self.update_traits_table(traits)

    def start_postprocess_thread(self):
        sender = self.sender()  # 返回发送当前信号的对象（控件）
        # 从控件获取参数
        args = {}
        args['is_denoise'] = self.ui.checkBox_is_denoise.isChecked()
        args['img_path'] = self.ui.label_DataDirShow_post.text()
        args['rsa'] = self.ui.checkBox_IsRSA.isChecked()
        args['dilation'] = self.ui.spinBox_dilate_iters.value()
        args['areathreshold'] = self.ui.spinBox_threshold_area.value()
        args['rba'] = self.ui.checkBox_IsRBA.isChecked()
        args['left'] = self.ui.spinBox_rba_left.value()
        args['right'] = self.ui.spinBox_rba_right.value()
        args['top'] = self.ui.spinBox_rba_top.value()
        args['bottom'] = self.ui.spinBox_rba_bottom.value()
        args['denoise_save_path'] = self.ui.lineEdit_SaveDir_denoise.text()
        # args['auto_iters'] = self.spinBox_AutoInpainting.value()
        if args['denoise_save_path'] == '':
            QMessageBox.warning(self, 'Warning', 'Please fill in save dir!')
            return
        inpaint_args = {}
        # inpaint_args['img_path'] = self.ui.label_DataDirShow_post.text()
        if sender == self.ui.btn_denoise:
            inpaint_args['is_inpaint'] = False
        else:
            inpaint_args['is_inpaint'] = self.ui.checkBox_is_inpaint.isChecked()
            inpaint_args['iters'] = self.ui.spinBox_inpaint_iters.value()
            inpaint_args['weight_path'] = self.ui.lineEdit_inpaint_weight_path.text()
            inpaint_args['inpaint_save_path'] = self.ui.lineEdit_SaveDir_post.text()
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
        # if sender == self.ui.btn_start_process:
        #     signals.img_info_signal.connect(self.start_calculate_thread)
        # 更新线程参数
        self.process_thread.args = args
        self.process_thread.inpaint_args = inpaint_args
        # 启动线程
        self.process_thread.start()

    def start_calculate_thread(self, img_file_path=None, img=None):
        args = {}
        args['save_path'] = self.ui.lineEdit_SaveDir_calculate.text()
        args['img_path'] = self.ui.label_DataDirShow_post.text()
        args['Layer_height'] = self.ui.spinBox_LayerHeight.value() if self.ui.spinBox_LayerHeight.value() else None
        args['Layer_width'] = self.ui.spinBox_LayerWidth.value() if self.ui.spinBox_LayerWidth.value() else None
        # if len(args['file_dict']) == 0:
        #     QMessageBox.warning(self, 'Warning', 'Please load image!')
        #     return
        if args['save_path'] == '':
            QMessageBox.warning(self, 'Warning', 'Please fill in save dir!')
            return
        # self.label_statubar.setText('Calculating...')
        # self.setabled('Calculate', False)
        # self.progressBar_statubar.setValue(0)
        # self.progressBar_statubar.setMaximum(num)
        self.calculate_thread = PhenotypeCalculateThread()
        # self.pushButton_StopCalculate.clicked.connect(self.calculate_thread.stop)
        self.calculate_thread.signal[str].connect(self.update_textBrowser)
        self.calculate_thread.signal[dict].connect(self.calculate_one_image)
        self.calculate_thread.show_img_signal.connect(self.update_graphicsView_Right)
        # self.calculate_thread.signal[str, str, str, np.ndarray].connect(self.finish_one_image)
        # self.calculate_thread.finished.connect(self.thread_finished)
        self.calculate_thread.args = args
        self.calculate_thread.img = img
        self.calculate_thread.img_file = img_file_path
        # self.calculate_thread.num = num
        self.calculate_thread.start()

    def start_all_process_thread(self, img_file_path, img):
        sender = self.sender()  # 返回发送当前信号的对象（控件）
        # 从控件获取参数
        denoise_args = {}
        denoise_args['is_denoise'] = self.ui.checkBox_is_denoise.isChecked()
        denoise_args['img_path'] = self.ui.label_DataDirShow_post.text()
        denoise_args['rsa'] = self.ui.checkBox_IsRSA.isChecked()
        denoise_args['dilation'] = self.ui.spinBox_dilate_iters.value()
        denoise_args['areathreshold'] = self.ui.spinBox_threshold_area.value()
        denoise_args['rba'] = self.ui.checkBox_IsRBA.isChecked()
        denoise_args['left'] = self.ui.spinBox_rba_left.value()
        denoise_args['right'] = self.ui.spinBox_rba_right.value()
        denoise_args['top'] = self.ui.spinBox_rba_top.value()
        denoise_args['bottom'] = self.ui.spinBox_rba_bottom.value()
        denoise_args['denoise_save_path'] = self.ui.lineEdit_SaveDir_denoise.text()
        # args['auto_iters'] = self.spinBox_AutoInpainting.value()
        if denoise_args['denoise_save_path'] == '':
            QMessageBox.warning(self, 'Warning', 'Please fill in save dir!')
            return

        inpaint_args = {}
        # inpaint_args['img_path'] = self.ui.label_DataDirShow_post.text()
        if sender == self.ui.btn_denoise:
            inpaint_args['is_inpaint'] = False
        else:
            inpaint_args['is_inpaint'] = self.ui.checkBox_is_inpaint.isChecked()
            inpaint_args['iters'] = self.ui.spinBox_inpaint_iters.value()
            inpaint_args['weight_path'] = self.ui.lineEdit_inpaint_weight_path.text()
            inpaint_args['inpaint_save_path'] = self.ui.lineEdit_SaveDir_post.text()
            if inpaint_args['inpaint_save_path'] == '':
                QMessageBox.warning(self, 'Warning', 'Please fill in save dir!')
                return

        calculate_args = {}
        calculate_args['calculate_save_path'] = self.ui.lineEdit_SaveDir_calculate.text()
        # calculate_args['img_path'] = self.ui.label_DataDirShow_post.text()
        calculate_args['Layer_height'] = self.ui.spinBox_LayerHeight.value() if self.ui.spinBox_LayerHeight.value() else None
        calculate_args['Layer_width'] = self.ui.spinBox_LayerWidth.value() if self.ui.spinBox_LayerWidth.value() else None
        if calculate_args['calculate_save_path'] == '':
            QMessageBox.warning(self, 'Warning', 'Please fill in save dir!')
            return
        # 初始化线程
        self.process_thread = PostProcessThread()
        # 定义线程信号和槽函数
        # self.ui.btn_Stop_post.clicked.connect(self.process_thread.stop)
        self.process_thread.process_signal.connect(self.update_textBrowser)
        self.process_thread.show_seg_img_signal.connect(self.update_graphicsView_post_Left)
        self.process_thread.show_post_img_signal.connect(self.update_graphicsView_Right)
        self.process_thread.trait_signal.connect(self.update_traits_table)

        # if sender == self.ui.btn_start_process:
        #     signals.img_info_signal.connect(self.start_calculate_thread)
        # 更新线程参数
        self.process_thread.img = img
        self.process_thread.image_path = img_file_path

        self.process_thread.denoise_args = denoise_args
        self.process_thread.inpaint_args = inpaint_args
        self.process_thread.calculate_args = calculate_args
        # 启动线程
        self.process_thread.start()

        
    def init_data(self):
        from config import default_cfg as cfg
        
        self.ui.radioButton_serving_post.setChecked(cfg.post_is_Serving)
        # 根系去噪
        self.ui.checkBox_is_denoise.setChecked(cfg.is_denoise)
        if cfg.is_denoise:
            self.ui.groupBox_imgDenoise.setEnabled(True)
        else:
            self.ui.groupBox_imgDenoise.setEnabled(False)
        self.ui.label_DataDirShow_post.setText(os.path.join(cfg.root_path, cfg.post_inputpath))
        self.ui.checkBox_IsRSA.setChecked(cfg.is_rsa)
        if cfg.is_rsa:
            self.ui.spinBox_dilate_iters.setEnabled(True)
            self.ui.spinBox_threshold_area.setEnabled(True)
        else:
            self.ui.spinBox_dilate_iters.setEnabled(False)
            self.ui.spinBox_threshold_area.setEnabled(False)
        self.ui.spinBox_dilate_iters.setValue(cfg.dilate_iters)
        self.ui.spinBox_threshold_area.setValue(cfg.threshold_area)

        self.ui.checkBox_IsRBA.setChecked(cfg.is_rba)
        if cfg.is_rba:
            self.ui.spinBox_rba_left.setEnabled(True)
            self.ui.spinBox_rba_right.setEnabled(True)
            self.ui.spinBox_rba_top.setEnabled(True)
            self.ui.spinBox_rba_bottom.setEnabled(True)
        else:
            self.ui.spinBox_rba_left.setEnabled(False)
            self.ui.spinBox_rba_right.setEnabled(False)
            self.ui.spinBox_rba_top.setEnabled(False)
            self.ui.spinBox_rba_bottom.setEnabled(False)
        self.ui.spinBox_rba_left.setValue(cfg.rba_left)
        self.ui.spinBox_rba_right.setValue(cfg.rba_right)
        self.ui.spinBox_rba_top.setValue(cfg.rba_top)
        self.ui.spinBox_rba_bottom.setValue(cfg.rba_bottom)
        
        self.ui.lineEdit_SaveDir_denoise.setText(os.path.join(cfg.root_path, cfg.denoise_savepath))

        # 根系修复
        self.ui.checkBox_is_inpaint.setChecked(cfg.is_inpaint)
        if cfg.is_inpaint:
            self.ui.groupBox_imgInpaint.setEnabled(True)
        else:
            self.ui.groupBox_imgInpaint.setEnabled(False)
        self.ui.spinBox_inpaint_iters.setValue(cfg.inpaint_iters)
        self.ui.cb_inpaint_runtime.addItems(cfg.inpaint_runtime_list)
        self.ui.cb_inpaint_runtime.setCurrentIndex(0)
        self.ui.cb_inpaint_model.addItems(cfg.inpaint_model_list)  # addItems和addItem的区别
        self.ui.cb_inpaint_model.setCurrentIndex(0)
        self.ui.lineEdit_inpaint_weight_path.setText(os.path.join(cfg.root_path, cfg.inpaint_weightpath))
        self.ui.lineEdit_SaveDir_post.setText(os.path.join(cfg.root_path, cfg.inpaint_savepath))

        # 性状计算
        self.ui.lineEdit_InputDir_calculate.setText(os.path.join(cfg.root_path, cfg.calculate_inputpath))
        self.ui.lineEdit_SaveDir_calculate.setText(os.path.join(cfg.root_path, cfg.calculate_savepath))
        # self.ui.spinBox_LayerHeight.setValue(cfg.layer_height)
        # self.ui.spinBox_LayerWidth.setValue(cfg.layer_width)

    def init_ui(self):
        ## 1.添加更新事件
        # 图像去噪
        self.ui.checkBox_is_denoise.clicked.connect(self.update_processing)
        self.ui.btn_DataDir_post.clicked.connect(self.update_select_dir_path)
        self.ui.radioButton_edge_post.clicked.connect(self.update_processing)
        self.ui.radioButton_serving_post.clicked.connect(self.update_processing)
        self.ui.checkBox_IsRSA.clicked.connect(self.update_processing)
        self.ui.checkBox_IsRBA.clicked.connect(self.update_processing)
        self.ui.spinBox_dilate_iters.valueChanged.connect(self.update_processing)
        self.ui.spinBox_threshold_area.valueChanged.connect(self.update_processing)
        self.ui.spinBox_rba_left.valueChanged.connect(self.update_processing)
        self.ui.spinBox_rba_right.valueChanged.connect(self.update_processing)
        self.ui.spinBox_rba_top.valueChanged.connect(self.update_processing)
        self.ui.spinBox_rba_bottom.valueChanged.connect(self.update_processing)
        self.ui.btn_SaveDir_denoise.clicked.connect(self.update_select_dir_path)

        # 图像修复
        self.ui.checkBox_is_inpaint.clicked.connect(self.update_processing)
        self.ui.spinBox_inpaint_iters.valueChanged.connect(self.update_processing)
        self.ui.btn_inpaint_weight_path.clicked.connect(self.update_select_file_path)
        self.ui.btn_SaveDir_post.clicked.connect(self.update_select_dir_path)

        # 性状计算
        self.ui.btn_InputDir_caculate.clicked.connect(self.update_select_dir_path)
        self.ui.btn_SaveDir_caculate.clicked.connect(self.update_select_dir_path)
        # self.ui.lineEdit_SaveDir_calculate.textChanged.connect(self.update_by_lineEdit)

        # self.pushButton_CalculateAll.clicked.connect(self.start_calculate)
        # self.pushButton_StopCalculate.clicked.connect(self.stop_thread)

        ## 2.添加线程事件
        # self.ui.btn_load_model.clicked.connect(self.load_model)  # 先加载模型
        self.ui.btn_denoise.clicked.connect(self.start_postprocess_thread)  # 图像去噪
        self.ui.btn_inpaint.clicked.connect(self.start_postprocess_thread)  # 图像修复
        self.ui.btn_start_analysis.clicked.connect(self.start_calculate_thread) # 性状计算
        self.ui.btn_start_process.clicked.connect(self.start_all_process_thread) # 全流程

        signals.img_postprocess_signal.connect(self.start_all_process_thread) # 全流程


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = ImgPostprocessWidget()
    window.show()
    
    sys.exit(app.exec_())
