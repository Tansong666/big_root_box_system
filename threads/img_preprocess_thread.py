from PyQt5.QtCore import *
from PyQt5.QtWidgets import QApplication
import sys
import os
import fastdeploy as fd
import cv2
import numpy as np
import time

# # 将项目根目录添加到 sys.path
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.insert(0, project_root)

# from threads.img_capture_thread import signal_auto_seg
from threads.img_save_thread import ImageSaveTask
from signals.global_signals import signals


class PreProcessThread(QThread):
    log_signal = pyqtSignal(str)  # 自定义信号，用于通知处理完成
    concat_signal = pyqtSignal(str,np.ndarray)  # 新增信号，用于通知拼接完成
    seg_signal = pyqtSignal(str,np.ndarray)

    def __init__(self):
        super(PreProcessThread, self).__init__()
        self.concat_args = None
        self.seg_args = None
        # self._is_running = False
        self.input_path = None
        self.model = None
        
        self.img1 = None
        self.img2 = None
        self.img1_path = None
        self.img2_path = None
        
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
                            # shape=(sum_rows, sum_cols, 3))  # 优化后：直接使用视图操作
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
            h, w = concat_matrix.shape[:2]
            window_h, window_w = slide_size
            stride_h, stride_w = slide_stride
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
        

    def img_preprocess(self,img1,img2,img1_path,img2_path,input_path=None,
                        is_concat=False,concat_savepath=None,x1=None,x2=None,x3=None,
                        is_seg=False,seg_savepath=None,is_slide=False,slide_size=None,slide_stride=None,is_resize=False,resize_scale=None): # 图像预处理
    # 边缘端和服务端 todo
        if input_path is None: # 传入的是图片
            self.log_signal.emit(f'开始进行自动前处理...')
    # try: todo
            img_name = str(img1_path.split("\\")[-2]) + ".png"
            image_save_path = None
            if is_concat: # 拼接
                start_time = time.time()
                img = self.concat(img1,img2,img1_path,img2_path,x1,x2,x3)
                end_time = time.time()
                elapsed_time = end_time - start_time
                self.log_signal.emit(f"图像拼接完成: {str(img.shape)}\n耗时: {elapsed_time:.2f}秒")
                
                image_save_path = os.path.join(concat_savepath, img_name)
                if not os.path.exists(os.path.dirname(image_save_path)):
                    os.makedirs(os.path.dirname(image_save_path))
                cv2.imwrite(image_save_path, img)
                self.log_signal.emit(f'{image_save_path} saved.')
                self.concat_signal.emit(image_save_path,img)
            if is_seg: # 图像分割
                start_time = time.time()
                img = self.seg(img,image_save_path,is_slide,slide_size,slide_stride,is_resize,resize_scale)
                # mask = self.auto_seg_v2(concat_matrix,self.code,self.args)  # 优化后
                end_time = time.time()
                elapsed_time = end_time - start_time
                self.log_signal.emit(f"图像分割成功: {str(img.shape)}\n耗时: {elapsed_time:.2f}秒")
                
                image_save_path = os.path.join(seg_savepath, img_name)
                if not os.path.exists(os.path.dirname(image_save_path)):
                    os.makedirs(os.path.dirname(image_save_path))
                cv2.imwrite(image_save_path, img)
                self.log_signal.emit(f'{image_save_path} saved.')
                self.seg_signal.emit(image_save_path, img)

                # # 发送信号，用于后续处理
                signals.img_postprocess_signal.emit(image_save_path,img)

               
            # except Exception as e:
            #     print(f"图像自动分割失败: {str(e)}")
            #     self.log_signal.emit(f"图像自动分割失败: {str(e)}")  
        else:
            # try:
                # 遍历拼接后的图像进行分割
                for root, _ , files in os.walk(input_path):
                    if not self._is_running:  # 检查是否需要停止线程
                        break
                    if files != [] and len(files) == 2:
                        img1_path=os.path.join(root,files[0])
                        img2_path=os.path.join(root,files[1]) 
                        # 记录开始时间
                        start_time = time.time()
                        img = self.concat(img1,img2,img1_path,img2_path,x1,x2,x3)
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        self.log_signal.emit(f"图像拼接完成: {str(img.shape)}\n耗时: {elapsed_time:.2f}秒")
                        
                        img_name = str(img1_path.split("\\")[-2]) + ".png"
                        image_save_path = os.path.join(concat_savepath, img_name)
                        if not os.path.exists(os.path.dirname(image_save_path)):
                            os.makedirs(os.path.dirname(image_save_path))
                        cv2.imwrite(image_save_path, img)
                        self.log_signal.emit(f'{image_save_path} saved.')
                        self.concat_signal.emit(image_save_path,img)

                        # 分割
                        start_time = time.time()
                        # code =os.path.join(root.split("\\")[-2], root.split("\\")[-1])
                        img = self.seg(img,image_save_path,is_slide,slide_size,slide_stride,is_resize,resize_scale)
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        self.log_signal.emit(f"图像分割成功: {str(img.shape)}\n耗时: {elapsed_time:.2f}秒")

                        image_save_path = os.path.join(seg_savepath, img_name)
                        if not os.path.exists(os.path.dirname(image_save_path)):
                            os.makedirs(os.path.dirname(image_save_path))
                        cv2.imwrite(image_save_path, img)
                        self.log_signal.emit(f'{image_save_path} saved.')
                        self.seg_signal.emit(image_save_path,img)

                self.log_signal.emit("全部图像分割完成")
                print("全部图像分割完成")
            # except Exception as e:
            #     print(f"图像分割失败: {str(e)}")
            #     self.log_signal.emit(f"图像分割失败: {str(e)}")
                
        # else:  # 非边缘检测
        #     pass

        # self._is_running = False

    def run(self):
        # print(self.concat_args)
        # print(self.seg_args)
        self.img_preprocess(self.img1, self.img2, self.img1_path, self.img2_path, 
                            self.input_path, **self.concat_args, **self.seg_args)
        # time.sleep(3)

    # def stop(self):
    #     self._is_running = False
    #     self.wait()  # 等待线程结束
    #     # self.quit()  # 退出线程

if __name__ == "__main__":
    app = QApplication(sys.argv)
    thread = PreProcessThread()
    thread.img1_path = "E:\\big_root_system\\data\\0102\\512\\1.png"
    thread.img2_path = "E:\\big_root_system\\data\\0102\\512\\2.png"
    thread.img1 = cv2.imread(thread.img1_path)
    thread.img2 = cv2.imread(thread.img2_path)
    thread.input_path = 'E:\\big_root_system\\data'
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
    thread.args = {
        'input_path': 'E:\\big_root_system\\data',
        'concat_save_path': 'E:\\big_root_system\\output\\concated',
        'seg_save_path': 'E:\\big_root_system\\output\\seg',
        'model_path': 'E:\\big_root_system\\models\\segment\\bigbox_segformer',
        'device': 'gpu',
        'use_trt': False,
        'use_paddle_trt': False,
        'slide_size': [512, 512],
        'stride': [400, 400]
    }
    # thread.start()
    sys.exit(app.exec_())

    # import os
    #
    # output_path = r'E:\big_root_system\output\concated'
    # if not os.access(os.path.dirname(output_path), os.W_OK):
    #     print("No write permission to the directory")