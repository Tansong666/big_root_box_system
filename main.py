"""
MainWindow -> ToolBar, MenuBar, StatusBar
1. 图像采集
2. 图像分割
3. 性状提取
"""

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import sys
from qt_material import apply_stylesheet

from ui.Ui_main_window import Ui_MainWindow
from sub_widgets.img_capture_widget import ImgCaptureWidget
from sub_widgets.img_seg_widget import ImgSegWidget
from sub_widgets.serial_assist_widget import SerialAssistWidget

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        # 创建对象
        self.ui = Ui_MainWindow()
        # 初始化内容
        self.ui.setupUi(self)
        # 初始化ui
        self.init_ui()

    def init_ui(self):
        # 创建一个 QWidget（这里使用 QLineEdit 作为示例）
        # min-width: 150px; /* 设置 tab 的最小宽度，可根据需要调整 */
        self.ui.tabWidget.tabBar().setStyleSheet("""
            QTabBar::tab {
                font-size: 8pt;
            }
        """)
        self.ui.tabWidget.addTab(ImgCaptureWidget(self), "根系图像采集")
        self.ui.tabWidget.addTab(ImgSegWidget(self), "根系图像分割")
        self.ui.tabWidget.addTab(SerialAssistWidget(self), "串口助手")

        self.ui.tabWidget.setCurrentIndex(0)
        # self.ui.tabWidget.setCurrentIndex(2)
        
        bar = self.statusBar()
        bar.showMessage("请选择功能")
        

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")  # 忽略所有警告

    app = QApplication(sys.argv)

    window = MainWindow()
    apply_stylesheet(app, theme='light_blue.xml')
    # apply_stylesheet(app, theme='dark_lightgreen.xml')
    
    window.show()
    sys.exit(app.exec_())

#     ['dark_amber.xml',
#  'dark_blue.xml',
#  'dark_cyan.xml',
#  'dark_lightgreen.xml',
#  'dark_pink.xml',
#  'dark_purple.xml',
#  'dark_red.xml',
#  'dark_teal.xml',
#  'dark_yellow.xml',
#  'light_amber.xml',
#  'light_blue.xml',
#  'light_cyan.xml',
#  'light_cyan_500.xml',
#  'light_lightgreen.xml',
#  'light_pink.xml',
#  'light_purple.xml',
#  'light_red.xml',
#  'light_teal.xml',
#  'light_yellow.xml']