from PyQt5.QtCore import QObject, pyqtSignal
import numpy as np

class GlobalSignals(QObject):
    # 定义一个全局信号，这里以发送字符串为例
    signal_auto_seg = pyqtSignal(np.ndarray, np.ndarray, str) # 用于自动分割图像信号
    signal_scaner_code = pyqtSignal(str)

# 创建全局信号实例
signals = GlobalSignals()