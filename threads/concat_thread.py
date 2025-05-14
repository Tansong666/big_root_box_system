import numpy as np
import cv2
import os
import sys
from PyQt5.QtCore import QThread, pyqtSignal, QObject, QFileSystemWatcher
from PyQt5.QtWidgets import QFileDialog, QApplication


class ImageMonitor(QObject):
    # 定义开始拼接的信号
    start_concat_signal = pyqtSignal(str, str, int, int, int)

    def __init__(self, target_folder, save_folder, x1=2000, x2=2340, x3=2730):
        super().__init__()
        self.target_folder = target_folder
        self.save_folder = save_folder
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3

        # 创建文件系统监控器
        self.watcher = QFileSystemWatcher()
        self.watcher.addPath(self.target_folder)

        # 连接信号和槽
        self.watcher.directoryChanged.connect(self.check_images)

    def check_images(self):
        # 检查文件夹中是否存在图片
        image_extensions = ['.png', '.jpg', '.jpeg']
        for root, dirs, files in os.walk(self.target_folder):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    # 存在图片，发出开始拼接的信号
                    self.start_concat_signal.emit(self.target_folder, self.save_folder, self.x1, self.x2, self.x3)
                    break


class ImgConcat_Thread(QThread):
    # 定义信号，可根据需要添加更多信号
    signal = pyqtSignal(str)

    def __init__(self, path_img, path_save, x1=2000, x2=2340, x3=2730):
        super(ImgConcat_Thread, self).__init__()
        self.path_img = path_img
        self.path_save = path_save
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.isOn = False

    def run(self):
        self.isOn = True
        self.signal.emit('开始拼接图像')
        try:
            for root, dirs, files in os.walk(self.path_img):
                if files != [] and len(files) == 2:
                    img1_path = os.path.join(root, files[0])
                    img2_path = os.path.join(root, files[1])
                    save_path1 = os.path.join(self.path_save, root.split("\\")[-3], root.split("\\")[-2][-3:] + '.png')
                    if os.path.exists(save_path1):
                        continue
                    else:
                        self.signal.emit(f'正在处理 {img1_path} 和 {img2_path}')
                        img2 = cv2.imread(img1_path)
                        img1 = cv2.imread(img2_path)
                        sum_rows = img1.shape[0]
                        sum_cols = img1.shape[1]
                        part1 = img1[0:sum_rows, 0:self.x1]
                        part2 = img1[0:sum_rows, self.x1:self.x2]
                        part3 = img1[0:sum_rows, self.x2:self.x3]
                        part4 = img1[0:sum_rows, self.x3:sum_cols]
                        part5 = img2[0:sum_rows, 0:self.x1]
                        part6 = img2[0:sum_rows, self.x1:self.x2]
                        part7 = img2[0:sum_rows, self.x2:self.x3]
                        part8 = img2[0:sum_rows, self.x3:sum_cols]
                        final_matrix = np.zeros((sum_rows, sum_cols, 3), np.uint8)
                        final_matrix[0:sum_rows, 0:self.x1] = part1
                        final_matrix[0:sum_rows, self.x1:self.x2] = part6
                        final_matrix[0:sum_rows, self.x2:self.x3] = part3
                        final_matrix[0:sum_rows, self.x3:sum_cols] = part8
                        save_dirs = os.path.join(self.path_save, root.split("\\")[-3])
                        if not os.path.isdir(save_dirs):
                            os.makedirs(save_dirs)
                        cv2.imwrite(save_path1, final_matrix, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                        self.signal.emit(f'已保存 {save_path1}')
        except Exception as e:
            self.signal.emit(f'处理过程中出现错误: {str(e)}')
        finally:
            self.isOn = False
            self.signal.emit('图像拼接完成')

    def stop(self):
        self.isOn = False


# if __name__ == '__main__':
#     path_img = r'H:\20240117rice\2021rice0117'
#     path_save = r'H:\20240117rice\0117pingjie'
#     thread = ConcatThread(path_img, path_save)
#     thread.signal.connect(lambda msg: print(msg))
#     thread.start()

if __name__ == '__main__':
    app = QApplication(sys.argv)

    # 定义监控文件夹和保存文件夹
    target_folder = r'H:\20240117rice\2021rice0117'
    save_folder = r'H:\20240117rice\0117pingjie'

    # 创建监控对象
    monitor = ImageMonitor(target_folder, save_folder)

    def start_concat(path_img, path_save, x1, x2, x3):
        # 创建拼接线程
        thread = ImgConcat_Thread(path_img, path_save, x1, x2, x3)
        thread.signal.connect(lambda msg: print(msg))
        thread.start()

    # 连接信号和槽
    monitor.start_concat_signal.connect(start_concat)

    sys.exit(app.exec_())