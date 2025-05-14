
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import sys
sys.path.append("../")

from ui.Ui_serial_setting_dialog import Ui_SerialSettingDialog

class SerialSettingDialog(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_SerialSettingDialog()
        self.ui.setupUi(self)

        # 可以通过此设置，固定对话框的大小
        self.setFixedSize(self.width(), self.height())
        self.initUi()

        self.baudrate = None
        self.data_bits = None

    def initUi(self):
        pass

    def accept(self):
        super().accept()
        # print("accept")
        # 读取当前设置值
        self.baudrate = self.ui.cb_baud_rate.currentText()
        self.data_bits = self.ui.cb_data.currentText()

    def reject(self):
        super().reject()
        # print("reject")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    dialog = SerialSettingDialog()
    dialog.show()
    sys.exit(app.exec_())