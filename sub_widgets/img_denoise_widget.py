# 系统模块
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import sys
import os

# 将项目根目录添加到 sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 自定义模块
# from common import utils
from ui.Ui_img_denoise_widget import Ui_DenoiseWidget


class ImgDenoiseWidget(QWidget):
    def __init__(self, parent=None):
        # 1. 初始化父类和ui
        super().__init__(parent)
        # self.main_window: QMainWindow = parent
        self.ui = Ui_DenoiseWidget()
        self.ui.setupUi(self)

        # 2.初始化数据
        # self.tcp_client = None

        # 3.初始化事件
        self.init_ui()

    def select_dir_path(self):
        dir_path = QFileDialog.getExistingDirectory(self, 'Open dir')
        if dir_path == '':
            return
        sender = self.sender()
        if sender == self.pushButton_SaveDir_train:
            self.lineEdit_SaveDir_train.setText(dir_path)
        elif sender == self.pushButton_SaveDir_predict:
            self.lineEdit_SaveDir_predict.setText(dir_path)
        elif sender == self.pushButton_DatasetPath:
            self.lineEdit_DatasetPath.setText(dir_path)
        elif sender == self.pushButton_ResumeModel:
            self.lineEdit_ResumeModel.setText(dir_path)
        elif sender == self.pushButton_DataDir:
            self.dataset_rootpath = dir_path
            self.label_DataDirShow.setText(dir_path)
            # self.file_list = []
            self.file_dict = {}
            self.treeWidget_Files.clear()
            self.load_treeWidget_from_dirpath(dir_path, self.treeWidget_Files)
            self.treeWidget_Files.expandAll()
            self.current_image = {'image_path': None, 'img': None, 'thresholdseg': None, 'gray': None,
                                  'processed': None, 'binary': None, 'traits': None, 'visualization': None}
            self.update_file_dict(self.lineEdit_SaveDir_predict)
            self.update_file_dict(self.lineEdit_SaveDir_postporcess)
            self.update_file_dict(self.lineEdit_SaveDir_calculate)
            self.update_current_image()
            self.inpainting_mode('quit')
        elif sender == self.pushButton_SaveDir_postporcess:
            self.lineEdit_SaveDir_postporcess.setText(dir_path)
        elif sender == self.pushButton_SaveDir_cacuate:
            self.lineEdit_SaveDir_calculate.setText(dir_path)
        elif sender == self.pushButton_SaveDir_warp:
            self.lineEdit_SaveDir_warp.setText(dir_path)
    
    def select_file_path(self):
        file_path = QFileDialog.getOpenFileName(self, 'Open file', './')
        if file_path[0] == '':
            return
        sender = self.sender()
        if sender == self.pushButton_WeightPath:
            self.lineEdit_WeighPath.setText(file_path[0])


    def init_ui(self):
        # 获取相关控件数据

        # 添加事件
        # self.ui.pushButton_DataDir
        self.ui.cb_mode.currentIndexChanged.connect(self.on_mode_changed)

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = ImgDenoiseWidget()
    window.show()
    
    sys.exit(app.exec_())
