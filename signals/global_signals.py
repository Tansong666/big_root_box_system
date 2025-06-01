from PyQt5.QtCore import QObject, pyqtSignal
import numpy as np

class GlobalSignals(QObject):
    # 定义一个全局信号，这里以发送字符串为例
    # img_preprocess_signal = pyqtSignal(str, str, np.ndarray, np.ndarray) # 用于自动拼接+分割图像信号
    # img_postprocess_signal = pyqtSignal(str, np.ndarray)
    img_process_singal = pyqtSignal(str, str, np.ndarray, np.ndarray)

    tab_change_signal = pyqtSignal(int) # 用于切换tab的信号
    
    # img_seg_signal = pyqtSignal(str, str, np.ndarray, np.ndarray)
    # code_scan_signal = pyqtSignal(str)

# 创建全局信号实例
signals = GlobalSignals()