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
from ui.Ui_img_process_widget import Ui_ImgProcessWidget
from threads.img_postprocess_thread import PostProcessThread
from threads.phenotype_calculate_thread import PhenotypeCalculateThread
from signals.global_signals import signals


class ImgProcessWidget(QWidget):
    def __init__(self, parent=None):
        # 1. 初始化父类和ui
        super().__init__(parent)
        # self.main_window: QMainWindow = parent
        self.ui = Ui_ImgProcessWidget()
        self.ui.setupUi(self)
        self.custom_Ui()

        # 2.初始化线程

        # 3.初始化控件数据
        self.init_data()

        # 4.初始化事件
        self.init_ui()

    def custom_Ui(self):
        # tab_calculate
        self.ui.trWItem_area = self.ui.trw_trait.topLevelItem(0)
        self.ui.trWItem_convex_area = self.ui.trw_trait.topLevelItem(1)
        self.ui.trWItem_length = self.ui.trw_trait.topLevelItem(2)
        self.ui.trWItem_diameter = self.ui.trw_trait.topLevelItem(3)
        self.ui.trWItem_depth = self.ui.trw_trait.topLevelItem(4)
        self.ui.trWItem_width = self.ui.trw_trait.topLevelItem(5)
        self.ui.trWItem_wdRatio = self.ui.trw_trait.topLevelItem(6)
        self.ui.trWItem_sturdiness = self.ui.trw_trait.topLevelItem(7)
        self.ui.trWItem_initial_x = self.ui.trw_trait.topLevelItem(8)
        self.ui.trWItem_initial_y = self.ui.trw_trait.topLevelItem(9)
        self.ui.trWItem_centroid_x = self.ui.trw_trait.topLevelItem(10)
        self.ui.trWItem_centroid_y = self.ui.trw_trait.topLevelItem(11)
        self.ui.trWItem_angle_apex_left = self.ui.trw_trait.topLevelItem(12)
        self.ui.trWItem_angle_apex_right = self.ui.trw_trait.topLevelItem(13)
        self.ui.trWItem_angle_apex_all = self.ui.trw_trait.topLevelItem(14)
        self.ui.trWItem_angle_entire_left = self.ui.trw_trait.topLevelItem(15)
        self.ui.trWItem_angle_entire_right = self.ui.trw_trait.topLevelItem(16)
        self.ui.trWItem_angle_entire_all = self.ui.trw_trait.topLevelItem(17)
        self.ui.trWItem_layer_mass = self.ui.trw_trait.topLevelItem(18)
        self.ui.trWItem_layer_mass.setExpanded(True)
        self.ui.trWItem_lmchild_Area = self.ui.trWItem_layer_mass.child(0)
        self.ui.trWItem_lmchild_Length = self.ui.trWItem_layer_mass.child(1)
        self.ui.trWItem_lmchild_Convex_hull = self.ui.trWItem_layer_mass.child(2)
        self.ui.trWItem_lmchild_A_C = self.ui.trWItem_layer_mass.child(3)
        self.ui.trWItem_lmchild_A_L = self.ui.trWItem_layer_mass.child(4)
        self.ui.trWItem_lmchild_L_C = self.ui.trWItem_layer_mass.child(5)

    def clear_logs(self):
        """清空日志窗口的内容"""
        self.ui.txtBrw_printer.clear()

    def update_textBrowser(self, text):
        if self.ui.splitter_view.sizes()[1] == 0:
            self.ui.splitter_view.setSizes([1, 1])
        if self.ui.splitter_right.sizes()[1] == 0:
            self.ui.splitter_right.setSizes([1, 1])
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        self.ui.txtBrw_printer.append(current_time + ' >>' + str(text))
        self.ui.txtBrw_printer.moveCursor(self.ui.txtBrw_printer.textCursor().End)
        self.ui.txtBrw_printer.repaint()

    def update_select_dir_path(self):
        dir_path = QFileDialog.getExistingDirectory(self, 'Open dir')
        if dir_path == '':
            return
        sender = self.sender()  # 返回发送当前信号的对象（控件）
        if sender == self.ui.btn_dataDirSelect:
            self.ui.lbl_dataDirShow.setText(dir_path)

        elif sender == self.ui.btn_concatSaveDir:
            self.ui.txt_concatSaveDir.setText(dir_path)

        elif sender == self.ui.btn_segWeightDirDir:
            self.ui.txt_segWeightDir.setText(dir_path)
        elif sender == self.ui.btn_segSaveDir:
            self.ui.txt_segSaveDir.setText(dir_path)

        elif sender == self.ui.btn_denoiseSaveDir:
            self.ui.txt_denoiseSaveDir.setText(dir_path)
        
        elif sender == self.ui.btn_inpaintWeightDir:
            self.ui.txt_inpaintWeightDir.setText(dir_path)
        elif sender == self.ui.btn_inpaintSaveDir:
            self.ui.txt_inpaintSaveDir.setText(dir_path)

        elif sender == self.ui.btn_calcuSaveDir:
            self.ui.txt_calcuSaveDir.setText(dir_path)
        elif sender == self.ui.btn_calcuInputDir:
            self.ui.txt_calcuInputDir.setText(dir_path)

    def update_select_file_path(self):
        file_path = QFileDialog.getOpenFileName(self, 'Open file', './')
        if file_path[0] == '':
            return
        sender = self.sender()
        if sender == self.ui.btn_inpaintWeightDir:
            self.ui.txt_inpaintWeightDir.setText(file_path[0])
        # elif sender == self.ui.btn_segWeightDir:
        #     self.ui.txt_segWeightDir.setText(file_path[0])

    def update_infer_mode(self):
        if self.ui.rdo_edge.isChecked():  
            self.infer_mode = 'edge' 
            self.ui.tab_imgSeg.setCurrentIndex(0)
            self.ui.tab2_servingSeg.setEnabled(False)
            self.ui.tab1_edgeSeg.setEnabled(True)
            self.ui.tab_imgDenoise.setCurrentIndex(0)
            self.ui.tab2_deeplearning.setEnabled(False)
            self.ui.tab1_opencv.setEnabled(True)
            self.ui.tab_imgInpaint.setCurrentIndex(0)  # 设置tabWidget_seg的当前索引为0
            self.ui.tab2_servingInpaint.setEnabled(False)
            self.ui.tab1_edgeInpaint.setEnabled(True)
        elif self.ui.rdo_serving.isChecked():  # 判断rdo_serving是否被选中
            self.infer_mode = 'serving'  # 如果被选中，将infer_mode设置为'serving'
            self.ui.tab_imgSeg.setCurrentIndex(1)  # 设置tabWidget_seg的当前索引为1
            self.ui.tab1_edgeSeg.setEnabled(False)
            self.ui.tab2_servingSeg.setEnabled(True)
            self.ui.tab_imgDenoise.setCurrentIndex(1)  # 设置tabWidget_seg的当前索引为1
            self.ui.tab1_opencv.setEnabled(False)
            self.ui.tab2_deeplearning.setEnabled(True)
            self.ui.tab_imgInpaint.setCurrentIndex(1)  
            self.ui.tab1_edgeInpaint.setEnabled(False)
            self.ui.tab2_servingInpaint.setEnabled(True)
        self.update_textBrowser('process_mode:'+ self.infer_mode)

    def update_preprocess(self):  # 拼接、分割
        sender = self.sender()
        # 推理模式
        # if sender == self.ui.rdo_edge or sender == self.ui.rdo_serving:
        # 拼接
        if sender == self.ui.chk_isConcat:
            if self.ui.chk_isConcat.isChecked():
                self.ui.grp_imgConcat.setEnabled(True)
            else:
                self.ui.grp_imgConcat.setEnabled(False)
            return
        if sender == self.ui.spin_x1:
            self.update_textBrowser('x1:'+ str(self.ui.spin_x1.value()))
            return
        if sender == self.ui.spin_x2:
            self.update_textBrowser('x2:'+ str(self.ui.spin_x2.value()))
            return
        if sender == self.ui.spin_x3:
            self.update_textBrowser('x3:'+ str(self.ui.spin_x3.value()))    
            return
        # 分割
        if sender == self.ui.chk_isSeg:
            if self.ui.chk_isSeg.isChecked():
                self.ui.grp_imgSeg.setEnabled(True)
            else:
                self.ui.grp_imgSeg.setEnabled(False)
            return
        if sender == self.ui.chk_isSlide:
            if self.ui.chk_isSlide.isChecked():
                self.ui.spin_cropSize.setEnabled(True)
                self.ui.spin_stride.setEnabled(True)
            else:
                self.ui.spin_cropSize.setEnabled(False)
                self.ui.spin_stride.setEnabled(False)
            return
        if sender == self.ui.chk_isResize:
            if self.ui.chk_isResize.isChecked():
                self.ui.dspin_scale.setEnabled(True)
            else:
                self.ui.dspin_scale.setEnabled(False)
            return
        if sender == self.ui.cb_segRunOpt:
            self.update_textBrowser('seg_run_opt:'+ str(self.ui.cb_segRunOpt.currentText()))
            return
        if sender == self.ui.cb_segModel:
            self.update_textBrowser('seg_model:'+ str(self.ui.cb_segModel.currentText()))
        if sender == self.ui.spin_cropSize:
            self.update_textBrowser('crop_size:'+ str(self.ui.spin_cropSize.value()))
            return
        if sender == self.ui.spin_stride:
            self.update_textBrowser('stride:'+ str(self.ui.spin_stride.value()))
            return
        if sender == self.ui.dspin_scale:
            self.update_textBrowser('scale:'+ str(self.ui.dspin_scale.value())) 
            return
    
    def update_postprocess(self):
        sender = self.sender()
        # 图像去噪
        if sender == self.ui.chk_isDenoise:
            if self.ui.chk_isDenoise.isChecked():
                self.ui.grp_imgDenoise.setEnabled(True)
            else:
                self.ui.grp_imgDenoise.setEnabled(False)
            return
        if sender == self.ui.chk_isRSA:
            if self.ui.chk_isRSA.isChecked():
                self.ui.spin_dilateIters.setEnabled(True)
                self.ui.spin_thresholdArea.setEnabled(True)
            else:
                self.ui.spin_dilateIters.setEnabled(False)
                self.ui.spin_thresholdArea.setEnabled(False)
            return
        if sender == self.ui.chk_isRBA:
            if self.ui.chk_isRBA.isChecked():
                self.ui.spin_leftPoint.setEnabled(True)
                self.ui.spin_rightPoint.setEnabled(True)
                self.ui.spin_topPoint.setEnabled(True)
                self.ui.spin_bottomPoint.setEnabled(True)
            else:
                self.ui.spin_leftPoint.setEnabled(False)
                self.ui.spin_rightPoint.setEnabled(False)
                self.ui.spin_topPoint.setEnabled(False)
                self.ui.spin_bottomPoint.setEnabled(False)
            return

        if sender == self.ui.spin_dilateIters:
            self.update_textBrowser('dilate_iters: ' + str(self.ui.spin_dilateIters.value()))
            return
        if sender == self.ui.spin_thresholdArea:
            self.update_textBrowser('threshold_area:'+ str(self.ui.spin_thresholdArea.value()))
            return
        if sender == self.ui.spin_leftPoint:
            self.update_textBrowser('rba_left:'+ str(self.ui.spin_leftPoint.value()))
            return
        if sender == self.ui.spin_rightPoint:
            self.update_textBrowser('rba_right:'+ str(self.ui.spin_rightPoint.value()))
            return
        if sender == self.ui.spin_topPoint:
            self.update_textBrowser('rba_top:'+ str(self.ui.spin_topPoint.value()))
            return
        if sender == self.ui.spin_bottomPoint:
            self.update_textBrowser('rba_bottom:'+ str(self.ui.spin_bottomPoint.value()))
            return

        # 根系修复
        if sender == self.ui.chk_isInpaint:
            if self.ui.chk_isInpaint.isChecked():
                self.ui.grp_imgInpaint.setEnabled(True)
            else:
                self.ui.grp_imgInpaint.setEnabled(False)
            return
        if sender == self.ui.spin_inpaintIters:
            self.update_textBrowser('inpaint_iters:'+ str(self.ui.spin_inpaintIters.value()))
            return
        if sender == self.ui.cb_inpaintRunOpt:
            self.update_textBrowser('inpaint_run_opt:'+ str(self.ui.cb_inpaintRunOpt.currentText()))
            return
        if sender == self.ui.cb_inpaintModel:
            self.update_textBrowser('inpaint_model:'+ str(self.ui.cb_inpaintModel.currentText()))
            return

        # 性状提取
        if sender == self.ui.chk_isCalculate:
            if self.ui.chk_isCalculate.isChecked():
                self.ui.tab_rootAnalysis.setEnabled(True)
            else:
                self.ui.tab_rootAnalysis.setEnabled(False)
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
        args['is_denoise'] = self.ui.chk_isDenoise.isChecked()
        args['img_path'] = self.ui.label_DataDirShow_post.text()
        args['rsa'] = self.ui.chk_isRSA.isChecked()
        args['dilation'] = self.ui.spin_dilateIters.value()
        args['areathreshold'] = self.ui.spin_thresholdArea.value()
        args['rba'] = self.ui.chk_isRBA.isChecked()
        args['left'] = self.ui.spin_leftPoint.value()
        args['right'] = self.ui.spin_rightPoint.value()
        args['top'] = self.ui.spin_topPoint.value()
        args['bottom'] = self.ui.spin_bottomPoint.value()
        args['denoise_save_path'] = self.ui.txt_denoiseSaveDir.text()
        # args['auto_iters'] = self.spinBox_AutoInpainting.value()
        if args['denoise_save_path'] == '':
            QMessageBox.warning(self, 'Warning', 'Please fill in save dir!')
            return
        inpaint_args = {}
        # inpaint_args['img_path'] = self.ui.label_DataDirShow_post.text()
        if sender == self.ui.btn_startDenoise:
            inpaint_args['is_inpaint'] = False
        else:
            inpaint_args['is_inpaint'] = self.ui.chk_isInpaint.isChecked()
            inpaint_args['iters'] = self.ui.spin_inpaintIters.value()
            inpaint_args['weight_path'] = self.ui.txt_inpaintWeightDir.text()
            inpaint_args['inpaint_save_path'] = self.ui.txt_inpaintSaveDir.text()
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
        # if sender == self.ui.btn_startProcess:
        #     signals.img_info_signal.connect(self.start_calculate_thread)
        # 更新线程参数
        self.process_thread.args = args
        self.process_thread.inpaint_args = inpaint_args
        # 启动线程
        self.process_thread.start()

    def start_calculate_thread(self, img_file_path=None, img=None):
        args = {}
        args['save_path'] = self.ui.txt_calcuSaveDir.text()
        args['img_path'] = self.ui.label_DataDirShow_post.text()
        args['Layer_height'] = self.ui.spin_layerHeight.value() if self.ui.spin_layerHeight.value() else None
        args['Layer_width'] = self.ui.spin_layerWidth.value() if self.ui.spin_layerWidth.value() else None
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
        denoise_args['is_denoise'] = self.ui.chk_isDenoise.isChecked()
        denoise_args['img_path'] = self.ui.label_DataDirShow_post.text()
        denoise_args['rsa'] = self.ui.chk_isRSA.isChecked()
        denoise_args['dilation'] = self.ui.spin_dilateIters.value()
        denoise_args['areathreshold'] = self.ui.spin_thresholdArea.value()
        denoise_args['rba'] = self.ui.chk_isRBA.isChecked()
        denoise_args['left'] = self.ui.spin_leftPoint.value()
        denoise_args['right'] = self.ui.spin_rightPoint.value()
        denoise_args['top'] = self.ui.spin_topPoint.value()
        denoise_args['bottom'] = self.ui.spin_bottomPoint.value()
        denoise_args['denoise_save_path'] = self.ui.txt_denoiseSaveDir.text()
        # args['auto_iters'] = self.spinBox_AutoInpainting.value()
        if denoise_args['denoise_save_path'] == '':
            QMessageBox.warning(self, 'Warning', 'Please fill in save dir!')
            return

        inpaint_args = {}
        # inpaint_args['img_path'] = self.ui.label_DataDirShow_post.text()
        if sender == self.ui.btn_startDenoise:
            inpaint_args['is_inpaint'] = False
        else:
            inpaint_args['is_inpaint'] = self.ui.chk_isInpaint.isChecked()
            inpaint_args['iters'] = self.ui.spin_inpaintIters.value()
            inpaint_args['weight_path'] = self.ui.txt_inpaintWeightDir.text()
            inpaint_args['inpaint_save_path'] = self.ui.txt_inpaintSaveDir.text()
            if inpaint_args['inpaint_save_path'] == '':
                QMessageBox.warning(self, 'Warning', 'Please fill in save dir!')
                return

        calculate_args = {}
        calculate_args['calculate_save_path'] = self.ui.txt_calcuSaveDir.text()
        # calculate_args['img_path'] = self.ui.label_DataDirShow_post.text()
        calculate_args['Layer_height'] = self.ui.spin_layerHeight.value() if self.ui.spin_layerHeight.value() else None
        calculate_args['Layer_width'] = self.ui.spin_layerWidth.value() if self.ui.spin_layerWidth.value() else None
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

        # if sender == self.ui.btn_startProcess:
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
        # 计算模式
        # self.ui.rdo_edge.setChecked(cfg.is_Edge)
        self.ui.rdo_serving.setChecked(cfg.is_Serving)

        # 数据路径
        self.ui.lbl_dataDirShow.setText(os.path.join(cfg.root_path, cfg.data_path))
        
        # 图像拼接参数
        self.ui.chk_isConcat.setChecked(cfg.is_concat)
        if cfg.is_concat:
            self.ui.grp_imgConcat.setEnabled(True)
        else:
            self.ui.grp_imgConcat.setEnabled(False)
        self.ui.spin_x1.setValue(cfg.concat_x1)
        self.ui.spin_x2.setValue(cfg.concat_x2)
        self.ui.spin_x3.setValue(cfg.concat_x3)
        self.ui.txt_concatSaveDir.setText(os.path.join(cfg.root_path, cfg.concat_savepath))
        
        # 根系分割参数
        self.ui.chk_isSeg.setChecked(cfg.is_seg)
        if cfg.is_seg:
            self.ui.grp_imgSeg.setEnabled(True)
        else:
            self.ui.grp_imgSeg.setEnabled(False)
        self.ui.cb_segRunOpt.addItems(cfg.seg_runtime_list)
        self.ui.cb_segRunOpt.setCurrentIndex(0)
        self.ui.cb_segModel.addItems(cfg.infer_model_list)  # addItems和addItem的区别
        self.ui.cb_segModel.setCurrentIndex(0)  # 注释这行代码，默认选择第一个选项
        self.ui.txt_segWeightDir.setText(os.path.join(cfg.root_path, cfg.seg_weightpath))
        self.ui.txt_segSaveDir.setText(os.path.join(cfg.root_path, cfg.seg_savepath))
        self.ui.chk_isSlide.setChecked(cfg.is_slide)
        self.ui.spin_cropSize.setValue(cfg.crop_size)
        self.ui.spin_stride.setValue(cfg.stride)
        self.ui.chk_isResize.setChecked(cfg.is_resize)
        self.ui.dspin_scale.setValue(cfg.resize_scale)
        
        # 根系去噪参数
        self.ui.chk_isDenoise.setChecked(cfg.is_denoise)
        if cfg.is_denoise:
            self.ui.grp_imgDenoise.setEnabled(True)
        else:
            self.ui.grp_imgDenoise.setEnabled(False)
        # self.ui.label_DataDirShow_post.setText(os.path.join(cfg.root_path, cfg.post_inputpath))
        self.ui.chk_isRSA.setChecked(cfg.is_rsa)
        if cfg.is_rsa:
            self.ui.spin_dilateIters.setEnabled(True)
            self.ui.spin_thresholdArea.setEnabled(True)
        else:
            self.ui.spin_dilateIters.setEnabled(False)
            self.ui.spin_thresholdArea.setEnabled(False)
        self.ui.spin_dilateIters.setValue(cfg.dilate_iters)
        self.ui.spin_thresholdArea.setValue(cfg.threshold_area)

        self.ui.chk_isRBA.setChecked(cfg.is_rba)
        if cfg.is_rba:
            self.ui.spin_leftPoint.setEnabled(True)
            self.ui.spin_rightPoint.setEnabled(True)
            self.ui.spin_topPoint.setEnabled(True)
            self.ui.spin_bottomPoint.setEnabled(True)
        else:
            self.ui.spin_leftPoint.setEnabled(False)
            self.ui.spin_rightPoint.setEnabled(False)
            self.ui.spin_topPoint.setEnabled(False)
            self.ui.spin_bottomPoint.setEnabled(False)
        self.ui.spin_leftPoint.setValue(cfg.rba_left)
        self.ui.spin_rightPoint.setValue(cfg.rba_right)
        self.ui.spin_topPoint.setValue(cfg.rba_top)
        self.ui.spin_bottomPoint.setValue(cfg.rba_bottom)
        
        self.ui.txt_denoiseSaveDir.setText(os.path.join(cfg.root_path, cfg.denoise_savepath))

        # 根系修复参数
        self.ui.chk_isInpaint.setChecked(cfg.is_inpaint)
        if cfg.is_inpaint:
            self.ui.grp_imgInpaint.setEnabled(True)
        else:
            self.ui.grp_imgInpaint.setEnabled(False)
        self.ui.spin_inpaintIters.setValue(cfg.inpaint_iters)
        self.ui.cb_inpaintRunOpt.addItems(cfg.inpaint_runtime_list)
        self.ui.cb_inpaintRunOpt.setCurrentIndex(0)
        self.ui.cb_inpaintModel.addItems(cfg.inpaint_model_list)  # addItems和addItem的区别
        self.ui.cb_inpaintModel.setCurrentIndex(0)
        self.ui.txt_inpaintWeightDir.setText(os.path.join(cfg.root_path, cfg.inpaint_weightpath))
        self.ui.txt_inpaintSaveDir.setText(os.path.join(cfg.root_path, cfg.inpaint_savepath))

        # 性状计算参数
        self.ui.chk_isCalculate.setChecked(cfg.is_calculate)
        if cfg.is_calculate:
            self.ui.tab_rootAnalysis.setEnabled(True)
        else:
            self.ui.tab_rootAnalysis.setEnabled(False)
        self.ui.txt_calcuInputDir.setText(os.path.join(cfg.root_path, cfg.calculate_inputpath))
        self.ui.txt_calcuSaveDir.setText(os.path.join(cfg.root_path, cfg.calculate_savepath))
        # self.ui.spin_layerHeight.setValue(cfg.layer_height)
        # self.ui.spin_layerWidth.setValue(cfg.layer_width)

    def init_ui(self):
        ## 1.添加控件更新事件
        # 计算模式
        self.ui.rdo_edge.clicked.connect(self.update_infer_mode) 
        self.ui.rdo_serving.clicked.connect(self.update_infer_mode)
        # 数据路径
        self.ui.btn_dataDirSelect.clicked.connect(self.update_select_dir_path)
        # 图像拼接
        self.ui.chk_isConcat.clicked.connect(self.update_preprocess)
        self.ui.spin_x1.valueChanged.connect(self.update_preprocess)  
        self.ui.spin_x2.valueChanged.connect(self.update_preprocess) 
        self.ui.spin_x3.valueChanged.connect(self.update_preprocess) 
        self.ui.btn_concatSaveDir.clicked.connect(self.update_select_dir_path)
        # 图像分割
        self.ui.chk_isSeg.clicked.connect(self.update_preprocess)
        self.ui.btn_segWeightDir.clicked.connect(self.update_select_dir_path)  # 模型文件夹
        self.ui.btn_segSaveDir.clicked.connect(self.update_select_dir_path)
        self.ui.cb_segRunOpt.currentIndexChanged.connect(self.update_preprocess)
        self.ui.cb_segModel.currentIndexChanged.connect(self.update_preprocess)
        self.ui.chk_isSlide.stateChanged.connect(self.update_preprocess)
        self.ui.chk_isResize.stateChanged.connect(self.update_preprocess)
        # self.ui.spin_cropSize.valueChanged.connect(self.update_preprocess)
        # self.ui.spin_stride.valueChanged.connect(self.update_preprocess)
        # self.ui.dspin_scale.valueChanged.connect(self.update_preprocess)

        # 图像去噪
        self.ui.chk_isDenoise.clicked.connect(self.update_postprocess)
        self.ui.chk_isRSA.clicked.connect(self.update_postprocess)
        self.ui.chk_isRBA.clicked.connect(self.update_postprocess)
        # self.ui.spin_dilateIters.valueChanged.connect(self.update_postprocess)
        # self.ui.spin_thresholdArea.valueChanged.connect(self.update_postprocess)
        # self.ui.spin_leftPoint.valueChanged.connect(self.update_postprocess)
        # self.ui.spin_rightPoint.valueChanged.connect(self.update_postprocess)
        # self.ui.spin_topPoint.valueChanged.connect(self.update_postprocess)
        # self.ui.spin_bottomPoint.valueChanged.connect(self.update_postprocess)
        self.ui.btn_denoiseSaveDir.clicked.connect(self.update_select_dir_path)
        # 图像修复
        self.ui.chk_isInpaint.clicked.connect(self.update_postprocess)
        self.ui.spin_inpaintIters.valueChanged.connect(self.update_postprocess)
        self.ui.btn_inpaintWeightDir.clicked.connect(self.update_select_file_path)  # 模型文件
        self.ui.btn_inpaintSaveDir.clicked.connect(self.update_select_dir_path)
        # 性状计算
        self.ui.chk_isCalculate.clicked.connect(self.update_postprocess)
        self.ui.btn_calcuInputDir.clicked.connect(self.update_select_dir_path)
        self.ui.btn_calcuSaveDir.clicked.connect(self.update_select_dir_path)
        # self.ui.txt_calcuSaveDir.textChanged.connect(self.update_by_lineEdit)

        ## 2.添加线程事件
        self.ui.btn_clearLog.clicked.connect(self.clear_logs)
        # self.ui.btn_loadSegModel.clicked.connect(self.load_model)  # 先加载模型
        # 拼接
        self.ui.btn_startConcat.clicked.connect(self.concat_thread)
        # 分割
        self.ui.btn_loadSegModel.clicked.connect(self.load_model)  # 先加载模型
        self.ui.btn_startSeg.clicked.connect(self.seg_thread)
        self.ui.btn_stopThread.clicked.connect(self.stop_thread)
        # 根据定义的信号连接槽函数
        signals.img_seg_signal.connect(self.auto_seg_thread)
        # 图像去噪
        self.ui.btn_startDenoise.clicked.connect(self.start_postprocess_thread)  # 图像去噪
        # 图像修复
        self.ui.btn_startInpaint.clicked.connect(self.start_postprocess_thread)  # 图像修复
        # 性状计算
        self.ui.btn_startCalculate.clicked.connect(self.start_calculate_thread) # 性状计算
        
        # 图像处理全流程
        self.ui.btn_startProcess.clicked.connect(self.start_all_process_thread) # 全流程
        signals.img_postprocess_signal.connect(self.start_all_process_thread) # 全流程


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = ImgProcessWidget()
    window.show()
    
    sys.exit(app.exec_())
