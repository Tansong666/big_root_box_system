from PyQt5.QtCore import *
import cv2
import numpy as np
import os
import time

from config.default_cfg import is_process
from drivers.camera import grab_image_save
from signals.global_signals import signals
from signals import global_vars


class ImgCapture_Thread(QThread):  # 用于扫描 这个类可能用于在一个新的线程中执行图像捕获任务。 信号可以用于将结果从线程发送到主线程。
    signal_img1_set = pyqtSignal(str, np.ndarray) # 自定义信号
    signal_img2_set = pyqtSignal(str, np.ndarray) # 信号可以用于将结果从线程发送到主线程。可能用于在图像捕获任务完成后通知其他对象。 # 用于扫描二维码信号
    # global com_arduino # 用于与arduino通信的串口
    # global com_scanning # 二维码扫描仪的串口
    # global direction # 方向 1为向前 2为向后 3为停止


    def __init__(self, savepath, startcode):
        super(ImgCapture_Thread, self).__init__()
        self.savapath = savepath  # 可能用于指定图像保存的路径
        self.startcode = startcode # 可能用于存储图像的编号
        self.is_process = False

    def run(self):  # 重写QThread的run方法，run方法在启动线程（start方法）后会被调用，用于执行线程的任务。
        if os.path.exists(self.savapath) == False:
            os.makedirs(self.savapath)

        scanning_index_last = 0
        scanning_index_now = 0
        while True:
            if global_vars.direction == 1: # 如果方向为1 则向前
                global_vars.com_arduino.write('5\r\n'.encode())  # 用于向arduino发送信号  serial.write() 方法用于向串行端口写入数据。它接受一个字符串作为参数，并将这个字符串发送到串行端口。
            elif global_vars.direction == 2:
                global_vars.com_arduino.write('6\r\n'.encode())
            else:
                global_vars.com_arduino.write('7\r\n'.encode())
                break  # 用于跳出循环

            while True:
                # scanning_index_last = scanning_index_now
                scanning_index_now = global_vars.com_scanning.read(3)
                scanning_index_now = scanning_index_now.decode('utf-8').strip() # 用于解码二维码扫描仪的数据
                # print(scanning_index_now)

                # if scanning_index_now != scanning_index_last and scanning_index_now != b'':
                if scanning_index_now != scanning_index_last and scanning_index_now != '': # 如果当前扫描位置与上一次扫描位置不同 且不为空
                    scanning_index_last = scanning_index_now
                    print(scanning_index_now)
                    break # 跳出循环
                elif global_vars.direction == 3:
                    break
            if self.startcode == scanning_index_now:
                self.savapath = self.savapath + f'_{self.startcode}'
            if not os.path.exists(self.savapath + '/' + scanning_index_now):
                os.makedirs(self.savapath + '/' + scanning_index_now)  # 递归地创建目录。如果目录已经存在，则抛出OSError。
            elif global_vars.direction == 3:
                global_vars.com_arduino.write('7\r\n'.encode())
                break
            # 计算漏拍
            dirs = os.listdir(self.savapath)
            numbers = [int(x) for x in dirs]
            numbers.sort()
            if numbers == [] or len(numbers) == 1:
                res = 1  # 用于计算漏拍，1为正常
                pass
            else:
                res = (max(numbers) - min(numbers)) / (len(numbers) - 1)  # 用于计算漏拍
            if res == 1:
                pass
            elif res != 1:
                global_vars.com_arduino.write('7\r\n'.encode())  # 用于向arduino发送停止信号
                # 删除文件夹
                os.removedirs(self.savapath + '/' + str(scanning_index_now))
                print('漏拍！！！')
                break
            # 开始拍摄
            global_vars.com_arduino.write('7\r\n'.encode())
            global_vars.com_arduino.write('1\r\n'.encode())  # 开灯

            img1 = grab_image_save()
            img1_save_path = os.path.join(self.savapath, scanning_index_now, str(1) + '.png')
            cv2.imwrite(img1_save_path, img1)
            self.signal_img1_set.emit(img1_save_path, img1)

            global_vars.com_arduino.write('2\r\n'.encode())

            img2 = grab_image_save()
            img2_save_path = os.path.join(self.savapath, scanning_index_now, str(2) + '.png')
            cv2.imwrite(img2_save_path, img2)
            self.signal_img2_set.emit(img2_save_path, img2)

            global_vars.com_arduino.write('3\r\n'.encode())

            signals.code_signal.emit(scanning_index_now)
            if self.is_process is True:
                # img1_save_path = "/home/ts/Documents/RootBoxSystem/RootBoxSystem_v1/data/root_data_demo/ori_captured/0510/171/1.png"
                # img2_save_path = "/home/ts/Documents/RootBoxSystem/RootBoxSystem_v1/data/root_data_demo/ori_captured/0510/171/2.png"
                # # img1 = cv2.imread(img1_save_path)
                # # img2 = cv2.imread(img2_save_path)
                # # signals.img_process_signal.emit(img1_save_path, img2_save_path, img1, img2)
                signals.img_process_path_signal.emit(img1_save_path,img2_save_path)
                print("开始处理该根系图像...")
            continue
