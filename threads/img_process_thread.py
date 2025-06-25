from PyQt5.QtCore import QThread, pyqtSignal,QThreadPool
import os
import cv2
import numpy as np
import time
import onnxruntime

# # 将项目根目录添加到 sys.path
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.insert(0, project_root)

from threads.img_save_thread import ImageSaveTask
# from signals.global_signals import signals
from threads.phenotype_calculate_thread import Calculater


class ImgProcessThread(QThread):
    process_signal = pyqtSignal(str)
    # 预处理相关信号
    concat_signal = pyqtSignal(str, np.ndarray)
    seg_signal = pyqtSignal(str, np.ndarray)
    # 后处理相关信号
    denosie_signal = pyqtSignal(str, np.ndarray)
    inpaint_signal = pyqtSignal(str, np.ndarray)
    calculate_signal = pyqtSignal(str, np.ndarray)
    trait_signal = pyqtSignal(dict)

    def __init__(self):
        super(ImgProcessThread, self).__init__()
        # 预处理参数
        # self.input_path = None
        self.model = None
        self.img1 = None
        self.img2 = None
        self.img1_path = None
        self.img2_path = None
        self.concat_args = None
        self.seg_args = None
        # 后处理参数
        self.denoise_args = None
        self.inpaint_args = None
        self.calculate_args = None

        # self._is_running = False
        # self.isOn = False
        
    def concat(self,img1,img2,img1_path,img2_path,x1=2000,x2=2700,x3=3350): # 自动拼接
        if img1 is None or img2 is None:
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)
        sum_rows = img1.shape[0]
        # the image length
        sum_cols = img1.shape[1]
        # part1 = img1[0:sum_rows, 0:x1]
        # part2 = img1[0:sum_rows, x1:x2]
        # part3 = img1[0:sum_rows, x2:x3]
        # part4 = img1[0:sum_rows, x3:sum_cols]
        # # 对F1图片进行剪切
        # part5 = img2[0:sum_rows, 0:x1]
        # part6 = img2[0:sum_rows, x1:x2]
        # part7 = img2[0:sum_rows, x2:x3]
        # part8 = img2[0:sum_rows, x3:sum_cols]
        final_matrix = np.zeros((sum_rows, sum_cols, 3), np.uint8)
        # final_matrix = np.memmap('temp_concat.dat', dtype=np.uint8, mode='w+',
        #                     shape=(sum_rows, sum_cols, 3))  # 优化后：直接使用视图操作
        # # 将需要的部分拼接起来
        final_matrix[0:sum_rows, 0:x1] = img2[0:sum_rows, 0:x1]  #part5
        final_matrix[0:sum_rows, x1:x2] = img1[0:sum_rows, x1:x2] # part2
        final_matrix[0:sum_rows, x2:x3] = img2[0:sum_rows, x2:x3]  # part7
        final_matrix[0:sum_rows, x3:sum_cols] = img1[0:sum_rows, x3:sum_cols] # part4
        # final_matrix[0:sum_rows, 0:x1] = part1
        # final_matrix[0:sum_rows, x1:x2] = part6
        # final_matrix[0:sum_rows, x2:x3] = part3
        # final_matrix[0:sum_rows, x3:sum_cols] = part8
        # print(finnal_save_path)
        # # 获取文件路径中的倒数第二个文件夹名称
        # code = img1_path.split("\\")[-2]
        # finnal_save_path = os.path.join(save_path, code + '.png')
        # self.concat_signal.emit(finnal_save_path, final_matrix)
        # self.log_signal.emit(f"图像拼接完成: {str(finnal_save_path)}{str(final_matrix.shape)}")
        # # 原代码：立即写入磁盘
        # # cv2.imwrite(finnal_save_path, final_matrix)
        # # 优化后：异步写入（需要添加QThreadPool）
        # QThreadPool.globalInstance().start(ImageSaveTask(finnal_save_path, final_matrix)) 
        return final_matrix     # 单次图片的拼接
        
    def seg(self,concat_matrix,concat_path,is_slide,slide_size,slide_stride,is_resize,resize_scale): # 自动分割
        if concat_matrix is None:
            concat_matrix = cv2.imread(concat_path,-1)
        if is_slide:
            # print(slide_size,slide_stride)
            h, w = concat_matrix.shape[:2]
            window_w, window_h = slide_size
            stride_w, stride_h = slide_stride
            # print(h,w,window_h,window_w,stride_h,stride_w)
            result_label_map = np.zeros((h, w), dtype=np.uint8)
            # 内存映射优化
            # result_label_map = np.memmap('temp_seg.dat', dtype=np.uint8, mode='w+', shape=(h, w))
            # Sliding window inference
            for y in range(0, h - window_h + 1, stride_h):
                # if not self._is_running:  # 检查是否需要停止线程
                #     break
                for x in range(0, w - window_w + 1, stride_w):
                    # if not self._is_running:  # 检查是否需要停止线程
                    #     break
                    window = concat_matrix[y:y + window_h, x:x + window_w]
                    window = cv2.cvtColor(window, cv2.COLOR_RGB2BGR)
                    window_result = self.model.predict(window)
                    # window_label_map = np.array(window_result.label_map).reshape(window_result.shape)
                    # result_label_map[y:y + window_h, x:x + window_w] = window_label_map.astype(np.uint8)
                    # 修改滑动窗口处理（减少临时变量）
                    result_label_map[y:y+window_h, x:x+window_w] = np.asarray(window_result.label_map).reshape(window_result.shape).astype(np.uint8)
            mask = np.where(result_label_map > 0, 255, 0).astype(np.uint8)
            # del result_label_map
        else:
            if is_resize:
                # 对图像进行等比例缩放
                concat_matrix = cv2.resize(concat_matrix, None, fx=resize_scale, fy=resize_scale)
            # 直接进行推理
            result = self.model.predict(concat_matrix)
            label_map = np.array(result.label_map).reshape(result.shape)
            mask = np.where(label_map > 0, 255, 0).astype(np.uint8)

        # # 保存掩码图
        # # image_path的文件名，包含后缀
        # image_name = os.path.basename(image_path)
        # finnal_seg_save_path = os.path.join(args['seg_save_path'],image_name)
        # if not os.path.exists(os.path.dirname(finnal_seg_save_path)):  # 判断文件夹是否存在
        #     os.makedirs(os.path.dirname(finnal_seg_save_path))  # 创建文件夹
        # # 发送分割完成信号
        # self.seg_signal.emit(finnal_seg_save_path,mask)    
        # # cv2.imwrite(finnal_seg_save_path, mask)
        # QThreadPool.globalInstance().start(ImageSaveTask(finnal_seg_save_path, mask))

        # self.log_signal.emit(f"图像分割成功: {finnal_seg_save_path}, mask.shape: {mask.shape}")

        return mask

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


    def inpaint(self, img, iters=None,weight_path='EUGAN.onnx',is_tensorrt=False):
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
        if is_tensorrt:
            import fastdeploy as fd
            from fastdeploy import ModelFormat
            option = fd.RuntimeOption()
            option.set_model_path(r"E:\big_root_system\models\inpaint\EUGAN.onnx", model_format=ModelFormat.ONNX)
            # **** GPU 配置 ***
            option.use_gpu(0)
            option.use_trt_backend()
            # option.set_trt_cache_file('EUGAN.trt')
            option.trt_option.serialize_file = 'EUGAN.trt'
            print("tensorrt加载成功")

            # 初始化构造runtime
            runtime = fd.Runtime(option)
            for i in range(iters):
                roi = runtime.infer({'input': roi})[0]
        else:
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
        

    def img_process(self,img1,img2,img1_path,img2_path,
                        is_concat=False,concat_savepath=None,x1=None,x2=None,x3=None,
                        is_seg=False,seg_savepath=None,is_slide=False,slide_size=None,slide_stride=None,is_resize=False,resize_scale=None,
                        is_denoise=False,denoise_savepath=None,rsa=False,dilation=None,areathreshold=None, rba=False, left=None, right=None, top=None, bottom=None,
                        is_inpaint=False,inpaint_savepath=None,iters=None,weight_path=None, 
                        is_calculate=False, calculate_savepath=None,Layer_height=None,Layer_width=None):
        self.process_signal.emit(f'开始进行自动处理...')
        # img_name = str(img1_path.split("/")[-2]) + ".png"  # linux
        img_name = str(img1_path.split("\\")[-2]) + ".png"  # windows
        image_save_path = None
        ## 执行图像拼接
        if is_concat:
            start_time = time.time()
            img = self.concat(img1,img2,img1_path,img2_path,x1,x2,x3)
            # 保存地址
            image_save_path = os.path.join(concat_savepath, img_name)
            if not os.path.exists(os.path.dirname(image_save_path)):
                os.makedirs(os.path.dirname(image_save_path))
            # 保存图像
            # cv2.imwrite(image_save_path, img)
            QThreadPool.globalInstance().start(ImageSaveTask(image_save_path, img))
            end_time = time.time()
            elapsed_time = end_time - start_time
            self.concat_signal.emit(image_save_path,img)
            self.process_signal.emit(f"图像拼接完成: {str(img.shape)}\n耗时: {elapsed_time:.2f}秒\n保存地址:{image_save_path}")
        ## 执行图像分割
        if is_seg: 
            start_time = time.time()
            img = self.seg(img,image_save_path,is_slide,slide_size,slide_stride,is_resize,resize_scale)
            # 保存地址
            image_save_path = os.path.join(seg_savepath, img_name)
            if not os.path.exists(os.path.dirname(image_save_path)):
                os.makedirs(os.path.dirname(image_save_path))
            # 保存图像
            # cv2.imwrite(image_save_path, img)
            QThreadPool.globalInstance().start(ImageSaveTask(image_save_path, img))
            end_time = time.time()
            elapsed_time = end_time - start_time
            self.seg_signal.emit(image_save_path, img)
            self.process_signal.emit(f"图像分割成功: {str(img.shape)}\n耗时: {elapsed_time:.2f}秒\n保存地址:{image_save_path}")
        ## 执行去噪处理
        if is_denoise:
            start_time = time.time()
            img = self.denoise(img,rsa,dilation,areathreshold,rba,left,right,top,bottom)
            # 保存地址
            image_save_path = os.path.join(denoise_savepath, img_name)
            if not os.path.exists(os.path.dirname(image_save_path)):
                os.makedirs(os.path.dirname(image_save_path))
            # 保存图像
            # cv2.imwrite(image_save_path, img)
            QThreadPool.globalInstance().start(ImageSaveTask(image_save_path, img))
            end_time = time.time()
            elapsed_time = end_time - start_time
            self.denosie_signal.emit(image_save_path,img)
            self.process_signal.emit(f"图像去噪成功: {str(img.shape)}\n耗时: {elapsed_time:.2f}秒\n保存地址:{image_save_path}")
        ## 执行图像修复
        if is_inpaint:
            start_time = time.time()
            img = self.inpaint(img, iters, weight_path)
            # 保存地址
            image_save_path = os.path.join(inpaint_savepath, img_name)
            if not os.path.exists(os.path.dirname(image_save_path)):
                os.makedirs(os.path.dirname(image_save_path))
            # 保存图像
            # cv2.imwrite(image_save_path, img)
            QThreadPool.globalInstance().start(ImageSaveTask(image_save_path, img))
            end_time = time.time()
            elapsed_time = end_time - start_time
            self.inpaint_signal.emit(image_save_path,img)
            self.process_signal.emit(f"图像修复成功: {str(img.shape)}\n耗时: {elapsed_time:.2f}秒\n保存地址:{image_save_path}")
        ## 执行特征计算
        if is_calculate:
            calculater = Calculater()
            show_img = calculater.loadimage(img, image_save_path)
            traits = calculater.get_traits(Layer_height=Layer_height, Layer_width=Layer_width)
            traits['image_path'] = image_save_path
            image_save_path = calculater.save_traits(calculate_savepath, img_name)
            self.calculate_signal.emit(image_save_path,show_img)
            self.trait_signal.emit(traits)
            self.process_signal.emit(f"图像特征计算成功: {str(img.shape)}\n保存地址:{image_save_path}")

    def run(self):
        self.img_process(self.img1, self.img2, self.img1_path, self.img2_path, 
                        **self.concat_args, **self.seg_args, 
                        **self.denoise_args, **self.inpaint_args, **self.calculate_args)

    # def stop(self):
    #     self._is_running = False
    #     self.wait()  # 等待线程结束
    #     # self.quit()  # 退出线程

if __name__ == "__main__":
    # 测试代码示例
    from PyQt5.QtWidgets import QApplication
    app = QApplication([])
    thread = ImgProcessThread()
    
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