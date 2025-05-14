from PyQt5.QtCore import *
import cv2
import numpy as np
import os

from drivers.camera import grab_image_save
from signals.global_signals import signals



class ImgCapture_Thread(QThread):  # 用于扫描 这个类可能用于在一个新的线程中执行图像捕获任务。 信号可以用于将结果从线程发送到主线程。
    signal_img1_set = pyqtSignal(np.ndarray) # 自定义信号
    signal_img2_set = pyqtSignal(np.ndarray) # 信号可以用于将结果从线程发送到主线程。可能用于在图像捕获任务完成后通知其他对象。 # 用于扫描二维码信号
    global com_arduino # 用于与arduino通信的串口
    global com_scanning # 二维码扫描仪的串口
    global direction # 方向 1为向前 2为向后 3为停止


    def __init__(self, savepath):
        super(ImgCapture_Thread, self).__init__()
        self.savapath = savepath  # 可能用于指定图像保存的路径

    def run(self):  # 重写QThread的run方法，run方法在启动线程（start方法）后会被调用，用于执行线程的任务。
        if os.path.exists(self.savapath) == False:
            os.makedirs(self.savapath)

        scanning_index_last = 0
        while True:
            if direction == 1: # 如果方向为1 则向前
                com_arduino.write('5\r\n'.encode())  # 用于向arduino发送信号  serial.write() 方法用于向串行端口写入数据。它接受一个字符串作为参数，并将这个字符串发送到串行端口。
            elif direction == 2:
                com_arduino.write('6\r\n'.encode())
            else:
                com_arduino.write('7\r\n'.encode())
                break  # 用于跳出循环

            while True:
                # scanning_index_last = scanning_index_now
                scanning_index_now = com_scanning.readline() # 用于读取二维码扫描仪的数据
                scanning_index_now = scanning_index_now.decode('utf-8').strip() # 用于解码二维码扫描仪的数据

                if scanning_index_now != scanning_index_last and scanning_index_now != '': # 如果当前扫描位置与上一次扫描位置不同 且不为空
                    scanning_index_last = scanning_index_now
                    print(scanning_index_now)
                    break # 跳出循环
                elif direction == 3:
                    break
            if os.path.exists(self.savapath + '/' + str(scanning_index_now)) == False:
                os.makedirs(self.savapath + '/' + str(scanning_index_now))  # 递归地创建目录。如果目录已经存在，则抛出OSError。
            elif direction == 3:
                com_arduino.write('7\r\n'.encode())
                break

            dirs = os.listdir(self.savapath)  # 用于获取指定路径下的文件和文件夹列表
            numbers = [int(x) for x in dirs]  # 用于将文件夹列表中的文件夹名转换为int类型
            numbers.sort()
            if numbers == [] or len(numbers) == 1: # 如果文件夹列表为空或者只有一个文件夹 则不进行计算
                res = 1  # 用于计算漏拍，1为正常
                pass
            else:
                res = (max(numbers) - min(numbers)) / (len(numbers) - 1)  # 用于计算漏拍
            if res == 1:
                pass
            elif res != 1:
                com_arduino.write('7\r\n'.encode())  # 用于向arduino发送停止信号
                # 删除文件夹
                os.removedirs(self.savapath + '/' + str(scanning_index_now))
                print('漏拍！！！')
                break

            com_arduino.write('7\r\n'.encode())
            # capture_qr_image(0, "qr_code") # 用于调用摄像头拍照 todo
            com_arduino.write('1\r\n'.encode())  # 用于向arduino发送信号 开灯
            img1 = grab_image_save() ##### 用于调用摄像头拍照
            cv2.imwrite(self.savapath + '/' + str(scanning_index_now) + '/' + str(1) + '.png', img1)
            # cv2.imwrite（filename ，img） filename 参数是你想要保存的文件的名称，包括文件路径和扩展名。img 参数是你想要保存的图像
            self.signal_img1_set.emit(img1)  # 当调用emit()方法时，信号对象会发出信号，并将img1作为参数传递给连接到该信号的槽（slot）函数。槽函数是与信号相关联的函数，它会在信号发出时被调用。
            com_arduino.write('2\r\n'.encode())
            img2 = grab_image_save() 
            cv2.imwrite(self.savapath + '/' + str(scanning_index_now) + '/' + str(2) + '.png', img2)

            self.signal_img2_set.emit(img2)
            com_arduino.write('3\r\n'.encode())
            # signals.signal_scaner_code.emit(str(scanning_index_now))
            signals.signal_auto_seg.emit(img1, img2, str(scanning_index_now))
            continue
