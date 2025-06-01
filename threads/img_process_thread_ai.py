from PyQt5.QtCore import QThread, pyqtSignal
import os
import cv2
import numpy as np
import time
import onnxruntime

from threads.img_save_thread import ImageSaveTask
from signals.global_signals import signals
from threads.phenotype_calculate_thread import Calculater

class ImageProcessThread(QThread):
    # 预处理相关信号
    log_signal = pyqtSignal(str)
    concat_signal = pyqtSignal(str, np.ndarray)
    seg_signal = pyqtSignal(str, np.ndarray)
    # 后处理相关信号
    process_signal = pyqtSignal(str)
    denosie_signal = pyqtSignal(str, np.ndarray)
    inpaint_signal = pyqtSignal(str, np.ndarray)
    calculate_signal = pyqtSignal(str, np.ndarray)
    trait_signal = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        # 预处理参数
        self.concat_args = None
        self.seg_args = None
        self.input_path = None
        self.model = None
        self.img1 = None
        self.img2 = None
        self.img1_path = None
        self.img2_path = None
        # 后处理参数
        self.denoise_args = None
        self.inpaint_args = None
        self.calculate_args = None

    # 预处理：图像拼接
    def concat(self, img1, img2, img1_path, img2_path, x1=2000, x2=2700, x3=3350):
        if img1 is None or img2 is None:
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)
        sum_rows = img1.shape[0]
        sum_cols = img1.shape[1]
        final_matrix = np.zeros((sum_rows, sum_cols, 3), np.uint8)
        final_matrix[0:sum_rows, 0:x1] = img2[0:sum_rows, 0:x1]
        final_matrix[0:sum_rows, x1:x2] = img1[0:sum_rows, x1:x2]
        final_matrix[0:sum_rows, x2:x3] = img2[0:sum_rows, x2:x3]
        final_matrix[0:sum_rows, x3:sum_cols] = img1[0:sum_rows, x3:sum_cols]
        return final_matrix

    # 预处理：图像分割
    def seg(self, concat_matrix, concat_path, is_slide, slide_size, slide_stride, is_resize, resize_scale):
        if concat_matrix is None:
            concat_matrix = cv2.imread(concat_path, -1)
        if is_slide:
            h, w = concat_matrix.shape[:2]
            window_h, window_w = slide_size
            stride_h, stride_w = slide_stride
            result_label_map = np.zeros((h, w), dtype=np.uint8)
            for y in range(0, h - window_h + 1, stride_h):
                for x in range(0, w - window_w + 1, stride_w):
                    window = concat_matrix[y:y + window_h, x:x + window_w]
                    window = cv2.cvtColor(window, cv2.COLOR_RGB2BGR)
                    window_result = self.model.predict(window)
                    result_label_map[y:y+window_h, x:x+window_w] = np.asarray(window_result.label_map).reshape(window_result.shape).astype(np.uint8)
            mask = np.where(result_label_map > 0, 255, 0).astype(np.uint8)
        else:
            if is_resize:
                concat_matrix = cv2.resize(concat_matrix, None, fx=resize_scale, fy=resize_scale)
            result = self.model.predict(concat_matrix)
            label_map = np.array(result.label_map).reshape(result.shape)
            mask = np.where(label_map > 0, 255, 0).astype(np.uint8)
        return mask

    # 后处理：去噪
    def denoise(self, img, rsa, dilation, areathreshold, rba, left, right, top, bottom):
        img = cv2.medianBlur(img, 5)
        if rsa:
            kernel = np.ones((3, 3), np.uint8) if dilation > 0 else None
            dilate = cv2.dilate(img, kernel, iterations=dilation) if dilation > 0 else img.copy()
            if areathreshold > 0:
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilate)
                for j in range(1, num_labels):
                    if stats[j][4] < areathreshold:
                        img[labels == j] = 0
        if rba:
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
            h, w = img.shape
            for j in range(1, num_labels):
                x, y = centroids[j]
                if x < left or x > w - right or y < top or y > h - bottom:
                    img[labels == j] = 0
        return img

    # 后处理：修复
    def inpaint(self, img, iters=None, weight_path='EUGAN.onnx'):
        if iters == 0:
            return img
        arr = np.where(img == 255)
        x_min, x_max = np.min(arr[1]), np.max(arr[1])
        y_min, y_max = np.min(arr[0]), np.max(arr[0])
        roi = img[y_min:y_max, x_min:x_max]
        roi = cv2.resize(roi, (384, 640))
        roi = cv2.threshold(roi, 127, 1, cv2.THRESH_BINARY)[1]
        roi = roi.astype(np.float32).reshape((1, 1, 640, 384))
        onnxmodel = onnxruntime.InferenceSession(weight_path)
        for _ in range(iters):
            roi = onnxmodel.run(None, {'input': roi})[0]
        roi = roi.squeeze()
        roi = cv2.resize(roi, (x_max - x_min, y_max - y_min))
        roi = cv2.threshold(roi, 0.75, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)
        output = np.pad(roi, ((y_min, img.shape[0]-y_max), (x_min, img.shape[1]-x_max)), 'constant')
        output = output * (1 - cv2.dilate(img/255, np.ones((3,3)), iterations=2)).astype(np.uint8)
        output = cv2.add(output, img)
        return output

    # 主处理流程：先预处理后后处理
    def img_process(self):
        # 执行预处理
        self.log_signal.emit('开始图像预处理...')
        concat_img = self.concat(self.img1, self.img2, self.img1_path, self.img2_path, **self.concat_args)
        seg_img = self.seg(concat_img, self.concat_args['concat_savepath'], **self.seg_args)
        
        # 保存预处理结果
        concat_savepath = os.path.join(self.concat_args['concat_savepath'], f"{os.path.basename(self.img1_path)}.png")
        cv2.imwrite(concat_savepath, concat_img)
        self.concat_signal.emit(concat_savepath, concat_img)
        
        seg_savepath = os.path.join(self.seg_args['seg_savepath'], f"{os.path.basename(self.img1_path)}.png")
        cv2.imwrite(seg_savepath, seg_img)
        self.seg_signal.emit(seg_savepath, seg_img)

        # 执行后处理
        self.process_signal.emit('开始图像后处理...')
        denoise_img = self.denoise(seg_img, **self.denoise_args)
        inpaint_img = self.inpaint(denoise_img, **self.inpaint_args)
        
        # 保存后处理结果
        denoise_savepath = os.path.join(self.denoise_args['denoise_savepath'], f"{os.path.basename(seg_savepath)}")
        cv2.imwrite(denoise_savepath, denoise_img)
        self.denosie_signal.emit(denoise_savepath, denoise_img)
        
        inpaint_savepath = os.path.join(self.inpaint_args['inpaint_savepath'], f"{os.path.basename(seg_savepath)}")
        cv2.imwrite(inpaint_savepath, inpaint_img)
        self.inpaint_signal.emit(inpaint_savepath, inpaint_img)

        # 特征计算
        if self.calculate_args.get('is_calculate'):
            calculater = Calculater()
            show_img = calculater.loadimage(inpaint_img, inpaint_savepath)
            traits = calculater.get_traits(**self.calculate_args)
            traits['image_path'] = inpaint_savepath
            calculate_savepath = calculater.save_traits(self.calculate_args['calculate_savepath'], os.path.basename(inpaint_savepath))
            self.calculate_signal.emit(calculate_savepath, show_img)
            self.trait_signal.emit(traits)

    def run(self):
        self.img_process()

if __name__ == "__main__":
    # 测试代码示例
    from PyQt5.QtWidgets import QApplication
    app = QApplication([])
    thread = ImageProcessThread()
    
    # 模拟输入参数
    thread.img1_path = "E:\\big_root_system\\data\\0102\\512\\1.png"
    thread.img2_path = "E:\\big_root_system\\data\\0102\\512\\2.png"
    thread.img1 = cv2.imread(thread.img1_path)
    thread.img2 = cv2.imread(thread.img2_path)
    
    thread.concat_args = {
        'is_concat': True,
        'concat_savepath': 'E:\\big_root_system\\output\\concated',
        'x1': 2000,
        'x2': 2700,
        'x3': 3350
    }
    
    thread.seg_args = {
        'is_seg': True,
        'seg_savepath': 'E:\\big_root_system\\output\\seg',
        'is_slide': True,
        'slide_size': [512, 512],
        'slide_stride': [400, 400],
        'is_resize': True,
        'resize_scale': 1.0
    }
    
    thread.denoise_args = {
        'is_denoise': True,
        'rsa': True,
        'dilation': 2,
        'areathreshold': 100,
        'rba': False,
        'denoise_savepath': 'E:\\big_root_system\\output\\denoised'
    }
    
    thread.inpaint_args = {
        'is_inpaint': True,
        'iters': 3,
        'weight_path': 'EUGAN.onnx',
        'inpaint_savepath': 'E:\\big_root_system\\output\\inpaint'
    }
    
    thread.calculate_args = {
        'is_calculate': True,
        'calculate_savepath': 'E:\\big_root_system\\output\\calculate',
        'Layer_height': 10,
        'Layer_width': 10
    }
    
    thread.start()
    app.exec_()