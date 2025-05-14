# 原代码：立即写入磁盘
# 优化后：异步写入（需要添加QThreadPool）
from PyQt5.QtCore import QRunnable, QThreadPool
import cv2
# import numpy as np

class ImageSaveTask(QRunnable):
    def __init__(self, path, matrix):
        super().__init__()
        self.path = path
        self.matrix = matrix
        
    def run(self):
        cv2.imwrite(self.path, self.matrix)

if __name__ == "__main__":
    # 示例用法
    finnal_save_path = "path/to/save/image.png"
    final_matrix = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)  # 示例矩阵
    QThreadPool.globalInstance().start(ImageSaveTask(finnal_save_path, final_matrix))