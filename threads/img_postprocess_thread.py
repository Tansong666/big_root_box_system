import os.path
import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
import onnxruntime
import time

from signals.global_signals import signals
from threads.phenotype_calculate_thread import Calculater

class PostProcessThread(QThread):
    # 定义信号
    # process_signal = pyqtSignal([str, str], [str])
    process_signal = pyqtSignal(str)
    show_seg_img_signal = pyqtSignal(np.ndarray)
    show_post_img_signal = pyqtSignal(np.ndarray)
    trait_signal = pyqtSignal(dict)
    # img_info_signal = pyqtSignal([str],[np.ndarray])

    def __init__(self):
        super(PostProcessThread, self).__init__()
        self.args = None  # 控件参数
        self.inpaint_args = None  # 控件参数
        self.calculate_args = None  # 控件参数
        # self.isOn = False
        self.img = None  # 信号发送参数
        self.code = None  # 信号发送参数

    def denoise(self,img,rsa,dilation,areathreshold,rba,left,right,top,bottom):
        ## 多次中值滤波 5x5
        img = cv2.medianBlur(img, 5)
        # img = cv2.medianBlur(img, 5)
        # img = cv2.medianBlur(img, 5)
        ## 多次中值滤波 3x3
        # img = cv2.medianBlur(img, 3)
        # img = cv2.medianBlur(img, 3)
        # img = cv2.medianBlur(img, 3)
        # img = cv2.medianBlur(img, 3)
        # img = cv2.medianBlur(img, 3)
        if rsa:
            if dilation > 0:
                kernel = np.ones((3, 3), np.uint8)
                dilate = cv2.dilate(img, kernel, iterations=dilation)
            else:
                dilate = img.copy()
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


    def inpaint(self, img, iters=None,weight_path='EUGAN.onnx'):
        if iters == 0:
            return img
        # 找到roi
        arr = np.where(img == 255)
        x_min, x_max = np.min(arr[1]), np.max(arr[1])
        y_min, y_max = np.min(arr[0]), np.max(arr[0])
        roi = img[y_min:y_max, x_min:x_max]
        roi = cv2.resize(roi, (384, 640))
        roi = cv2.threshold(roi, 127, 1, cv2.THRESH_BINARY)[1]
        roi.resize((1, 1, 640, 384))
        roi = roi.astype(np.float32)

        onnxmodel = onnxruntime.InferenceSession(weight_path)

        for i in range(iters):
            roi = onnxmodel.run(None, {'input': roi})[0]
        roi = roi.squeeze().copy()
        roi = cv2.resize(roi, (x_max - x_min, y_max - y_min))
        roi = cv2.threshold(roi, 0.75, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)
        output = np.pad(roi, ((y_min, img.shape[0] - y_max), (x_min, img.shape[1] - x_max)), 'constant')
        output = output * (1 - cv2.dilate(img / 255, np.ones((3, 3), np.uint8), iterations=2))
        output = output.astype(np.uint8)
        output_mask = cv2.dilate(output, np.ones((3, 3), np.uint8), iterations=3)
        output = cv2.add(output, img)
        output = cv2.morphologyEx(output, cv2.MORPH_CLOSE, np.ones((7, 7)), iterations=1)
        output = cv2.dilate(output, np.ones((3, 3)), iterations=1)
        output = cv2.blur(output, (3, 3))
        output = cv2.erode(output, np.ones((3, 3)), iterations=1)
        output = output * (output_mask // 255)
        output = cv2.dilate(output, np.ones((3, 3)), iterations=3)
        output = cv2.blur(output, (9, 9))
        output = cv2.erode(output, np.ones((3, 3)), iterations=2)
        output = cv2.threshold(output, 127, 255, cv2.THRESH_BINARY)[1]
        output = cv2.add(output, img)
        return output


    def img_postprocess(self,img, code, 
                    is_denoise=False, img_path=None, rsa=False,dilation=None, areathreshold=None, rba=False, left=None, right=None, top=None, bottom=None,denoise_save_path=None,
                    is_inpaint=False, iters=None,weight_path=None,inpaint_save_path=None, 
                    calculate_save_path=None,Layer_height=None,Layer_width=None):

        self.process_signal.emit(f'开始进行后处理...')
        # print(f'Processing images...')
        if img is None:
            # 批量处理图片路径中的所有图片
            for root, dirs, files in os.walk(img_path):
                # 构造与输入目录结构一致的输出子目录
                relative_path = os.path.relpath(root, img_path)
                denoise_subdir = os.path.join(denoise_save_path, relative_path)
                # inpaint_subdir = os.path.join(inpaint_save_path, relative_path)
                
                for file in files:
                    # 仅处理图片文件（可根据需求扩展格式）
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        img_file = os.path.join(root, file)
                        self.process_signal.emit(f'正在处理该图片: {img_file}')
                        # 读取图片（假设是灰度图，根据实际情况调整）
                        # img = cv2.imread(img_file, 0)
                        img = cv2.imread(img_file, -1)  # -1表示读取原始图像，不进行转化，包括alpha通道
                        self.show_seg_img_signal.emit(img)
                        if img is None:
                            # self.process_signal.emit(f'Warning: 无法读取图片 {img_file}')
                            print(f'Warning: 无法读取图片 {img_file}')
                            continue
                        ## 执行去噪处理
                        if is_denoise:
                            img = self.denoise(img, rsa, dilation, areathreshold, rba, left, right, top, bottom)
                            os.makedirs(denoise_subdir, exist_ok=True)
                            # 构造输出路径（保持原文件名）
                            image_save_path = os.path.join(denoise_subdir, file)
                            cv2.imwrite(image_save_path, img)
                            self.process_signal.emit(f'根系除杂: <{image_save_path}>saved')
                            self.show_post_img_signal.emit(img)
                            # 睡眠1秒
                            # time.sleep(1)
                        ## 执行图像修复
                        if is_inpaint:
                            img = self.inpaint(img, iters, weight_path)
                            # 创建输出目录（如果不存在）
                            inpaint_subdir = os.path.join(inpaint_save_path, relative_path)
                            os.makedirs(inpaint_subdir, exist_ok=True)
                            # 构造输出路径（保持原文件名）
                            image_save_path = os.path.join(inpaint_subdir, file)
                            cv2.imwrite(image_save_path, img)
                            # 发送信号
                            self.process_signal.emit(f'根系修复: <{image_save_path}>saved')
                            self.show_post_img_signal.emit(img)
                            # 睡眠1秒
                            # time.sleep(1)

                        calculater = Calculater()
                        show_img = calculater.loadimage(img, image_save_path)
                        self.show_post_img_signal.emit(show_img)
                        traits = calculater.get_traits(Layer_height=Layer_height, Layer_width=Layer_width)
                        traits['image_path'] = image_save_path
                        self.trait_signal.emit(traits)
                        calculater.save_traits(calculate_save_path, file)       
                        # signals.img_info_signal.emit(image_save_path, img)
            self.process_signal.emit(f'根系后处理全部完成.')
        else:
            self.show_seg_img_signal.emit(img)
            ## 执行去噪处理
            if is_denoise:
                img = self.denoise(img,rsa,dilation,areathreshold,rba,left,right,top,bottom)
                image_save_path = os.path.join(denoise_save_path, code + '.png')
                if not os.path.exists(os.path.dirname(image_save_path)):
                    os.makedirs(os.path.dirname(image_save_path))
                cv2.imwrite(image_save_path, img)
                self.process_signal.emit(f'{image_save_path} saved.')
                self.show_post_img_signal.emit(img)
            ## 执行图像修复
            if is_inpaint:
                img = self.inpaint(img, iters, weight_path)
                image_save_path = os.path.join(denoise_save_path, code + '.png')
                if not os.path.exists(os.path.dirname(image_save_path)):
                    os.makedirs(os.path.dirname(image_save_path))
                cv2.imwrite(image_save_path, img)
                self.process_signal.emit(f'{image_save_path} saved.')
                self.show_post_img_signal.emit(img)

            calculater = Calculater()
            show_img = calculater.loadimage(img, image_save_path)
            self.show_post_img_signal.emit(show_img)
            traits = calculater.get_traits(Layer_height=Layer_height, Layer_width=Layer_width)
            traits['image_path'] = image_save_path
            self.trait_signal.emit(traits)
            file = code + '.png'
            calculater.save_traits(calculate_save_path, file)

            # signals.img_info_signal.emit(image_save_path, img)
        
        # return img


    def run(self):
        # self.isOn = True
        # denoise_img = self.img_denoise(self.img, self.code, **self.args)
        # self.img_inpaint(denoise_img, **self.inpaint_args)
        self.img_postprocess(self.img, self.code, **self.args, **self.inpaint_args, **self.calculate_args)

        # self.isOn = False

    # def stop(self):
    #     self.isOn = False
        # self.terminate()
        # self.wait()
        # self.deleteLater()
        



if __name__ == '__main__':
    process = PostProcessThread()
    args = {
        'img_path': r'E:\big_root_system\output\segmented',
        'rso': True,
        'dilation': 0,
        'areathreshold': 100,
        'rbo': False,
        'left': 0,
        'right': 0,
        'top': 0,
        'bottom': 0,
        'denoise_save_path': r'E:\big_root_system\output\denoised'
    }
    process.args = args
    process.run()
