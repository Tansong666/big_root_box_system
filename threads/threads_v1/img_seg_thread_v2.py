from PyQt5.QtCore import *
from PyQt5.QtWidgets import QApplication
import sys
import os
import fastdeploy as fd
import cv2
import numpy as np
import time

# from threads.img_capture_thread import signal_auto_seg
from threads.img_save_thread import ImageSaveTask
from signals.global_signals import signals


# # 将项目根目录添加到 sys.path
# project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.insert(0, project_root)

class ImgSegThread(QThread):
    signal_process_complete = pyqtSignal(str)  # 自定义信号，用于通知处理完成
    signal_concat_complete = pyqtSignal(str,np.ndarray)  # 新增信号，用于通知拼接完成
    signal_seg_complete = pyqtSignal(str,np.ndarray)

    def __init__(self):
        super(ImgSegThread, self).__init__()
        self.args = {}
        self.model = None
        self._is_running = False
        self.img1 = None
        self.img2 = None
        self.img1_path = None
        self.img2_path = None
        

    def one_concat(self,root_path,files_path,save_path,x1=2000,x2=2700,x3=3350): # 单次图片的拼接
        img1_path=os.path.join(root_path,files_path[0])
        img2_path=os.path.join(root_path,files_path[1]) 
        finnal_save_path = os.path.join(save_path, root_path.split("\\")[-2], root_path.split("\\")[-1] + '.png')
        if not os.path.exists(os.path.dirname(finnal_save_path)):  # 判断文件夹是否存在
            os.makedirs(os.path.dirname(finnal_save_path))  # 创建文件夹
        # if os.path.exists(finnal_save_path): # 判断文件是否存在
        #     pass
        # else:
        img1 = cv2.imread(img1_path)  # 读取图片       img2 = cv2.imread(path2)
        img2 = cv2.imread(img2_path)
        # img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # img2 = cv2.rotate(img2, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # the image height
        sum_rows = img1.shape[0]
        # the image length
        sum_cols = img1.shape[1]
        # 对F0图片进行剪切
        # x1 = 2000  # 第一刀
        # x2 = 2680  # 第二刀
        # x3 = 3350  # 第三刀
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
        # final_matrix = np.memmap('temp_concat_one.dat', dtype=np.uint8, mode='w+', 
        #                     shape=(sum_rows, sum_cols, 3))  # 优化后：直接使用视图操作
        # # 将需要的部分拼接起来
        final_matrix[0:sum_rows, 0:x1] = img2[0:sum_rows, 0:x1]  #part5
        final_matrix[0:sum_rows, x1:x2] = img1[0:sum_rows, x1:x2] # part2
        final_matrix[0:sum_rows, x2:x3] = img2[0:sum_rows, x2:x3]  # part7
        final_matrix[0:sum_rows, x3:sum_cols] = img1[0:sum_rows, x3:sum_cols] # part4
        # print(finnal_save_path)
        self.signal_concat_complete.emit(finnal_save_path, final_matrix)
        self.signal_process_complete.emit(f"图像拼接完成: {str(finnal_save_path)}{str(final_matrix.shape)}")
        # cv2.imwrite(finnal_save_path, final_matrix)
        QThreadPool.globalInstance().start(ImageSaveTask(finnal_save_path, final_matrix)) 
    
        return finnal_save_path, final_matrix        # 单次图片的拼接

    def auto_concat(self,img1,img2,img1_path,img2_path,save_path,x1=2000,x2=2700,x3=3350): # 自动拼接
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
        # 获取文件路径中的倒数第二个文件夹名称
        code = img1_path.split("\\")[-2]
        finnal_save_path = os.path.join(save_path, code + '.png')
        self.signal_concat_complete.emit(finnal_save_path, final_matrix)
        self.signal_process_complete.emit(f"图像拼接完成: {str(finnal_save_path)}{str(final_matrix.shape)}")
        # 原代码：立即写入磁盘
        # cv2.imwrite(finnal_save_path, final_matrix)
        # 优化后：异步写入（需要添加QThreadPool）
        QThreadPool.globalInstance().start(ImageSaveTask(finnal_save_path, final_matrix)) 
    
        return finnal_save_path, final_matrix        # 单次图片的拼接

    def auto_seg_v2(self,concat_matrix,code,args): # 自动分割
        if args['slide_predict']:
            from concurrent.futures import ThreadPoolExecutor
            h, w = concat_matrix.shape[:2]
            window_h, window_w = args['slide_size']
            stride_h, stride_w = args['stride']
            # 生成唯一的内存映射文件名，避免冲突
            import uuid
            temp_file = f"temp_seg_{uuid.uuid4().hex}.dat"
            result_label_map = np.memmap(temp_file, dtype=np.uint8, mode='w+', shape=(h, w))

            def process_window(y, x):
                if not self._is_running:
                    return
                window = concat_matrix[y:y + window_h, x:x + window_w]
                # 避免使用 cv2.cvtColor，直接操作数组
                window = cv2.cvtColor(window, cv2.COLOR_RGB2BGR)
                window_result = self.model.predict(window)
                result_label_map[y:y+window_h, x:x+window_w] = np.asarray(window_result.label_map).reshape(window_result.shape).astype(np.uint8)

            # 使用线程池并行处理窗口
            with ThreadPoolExecutor() as executor:
                futures = []
                for y in range(0, h - window_h + 1, stride_h):
                    for x in range(0, w - window_w + 1, stride_w):
                        futures.append(executor.submit(process_window, y, x))
                # 检查是否需要停止线程
                for future in futures:
                    if not self._is_running:
                        break
                    future.result()

            mask = np.where(result_label_map > 0, 255, 0).astype(np.uint8)
            # 删除内存映射文件
            del result_label_map
            try:
                os.remove(temp_file)
            except Exception as e:
                print(f"删除临时文件 {temp_file} 失败: {e}")
        else:
            if args['resize_predict']:
                # 对图像进行等比例缩放
                im = cv2.resize(concat_matrix, None, fx=args['resize_scale'], fy=args['resize_scale'])
            # 直接进行推理
            result = self.model.predict(im)
            label_map = np.array(result.label_map).reshape(result.shape)
            mask = np.where(label_map > 0, 255, 0).astype(np.uint8)

        # 发送分割完成信号
        self.signal_seg_complete.emit(mask)

        # 保存掩码图
        finnal_seg_save_path = os.path.join(args['seg_save_path'], code + '.png')
        if not os.path.exists(os.path.dirname(finnal_seg_save_path)):  # 判断文件夹是否存在
            os.makedirs(os.path.dirname(finnal_seg_save_path))  # 创建文件夹
        # cv2.imwrite(finnal_seg_save_path, mask)
        QThreadPool.globalInstance().start(ImageSaveTask(finnal_seg_save_path, mask))

        self.signal_process_complete.emit(f"图像分割成功: {finnal_seg_save_path}")

        return mask
        
    def auto_seg(self,concat_matrix,image_path,args): # 自动分割
        if args['slide_predict']:
            h, w = concat_matrix.shape[:2]
            window_h, window_w = args['slide_size']
            stride_h, stride_w = args['stride']
            result_label_map = np.zeros((h, w), dtype=np.uint8)
            # 内存映射优化
            # result_label_map = np.memmap('temp_seg.dat', dtype=np.uint8, mode='w+', shape=(h, w))
            # Sliding window inference
            for y in range(0, h - window_h + 1, stride_h):
                if not self._is_running:  # 检查是否需要停止线程
                    break
                for x in range(0, w - window_w + 1, stride_w):
                    if not self._is_running:  # 检查是否需要停止线程
                        break
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
            if args['resize_predict']:
                # 对图像进行等比例缩放
                im = cv2.resize(concat_matrix, None, fx=args['resize_scale'], fy=args['resize_scale'])
            # 直接进行推理
            result = self.model.predict(im)
            label_map = np.array(result.label_map).reshape(result.shape)
            mask = np.where(label_map > 0, 255, 0).astype(np.uint8)

        # 保存掩码图
        # image_path的文件名，包含后缀
        image_name = os.path.basename(image_path)
        finnal_seg_save_path = os.path.join(args['seg_save_path'],image_name)
        if not os.path.exists(os.path.dirname(finnal_seg_save_path)):  # 判断文件夹是否存在
            os.makedirs(os.path.dirname(finnal_seg_save_path))  # 创建文件夹
        # 发送分割完成信号
        self.signal_seg_complete.emit(finnal_seg_save_path,mask)    
        # cv2.imwrite(finnal_seg_save_path, mask)
        QThreadPool.globalInstance().start(ImageSaveTask(finnal_seg_save_path, mask))

        self.signal_process_complete.emit(f"图像分割成功: {finnal_seg_save_path}, mask.shape: {mask.shape}")

        return finnal_seg_save_path, mask
        

    def run(self):
        self._is_running = True
        # img1 = self.args.get('img1', None)  # 用于获取图像1
        # img2 = self.args.get('img2', None)  # 用于获取图像2
        # code = self.args.get('code', None)  # 用于获取图像2
        is_auto_seg = self.args.get('auto_seg', False)  # 用于获取图像2
        # get函数的第一个参数是键名，第二个参数是默认值
        if self.args.get('is_edge', True):  # 判断是否是边缘检测
            # # runtime_option参数
            # device = self.args.get('device', 'gpu')
            # use_trt = self.args.get('use_trt', False)
            # use_paddle_trt = self.args.get('use_paddle_trt', False)
            # # 构建模型参数
            # model_path = self.args.get('model_path') 
            # 图像拼接参数
            img_path = self.args.get('img_path')
            concat_save_path = self.args.get('concat_save_path')
            x1 = self.args.get('x1', 2000)
            x2 = self.args.get('x2', 2340)
            x3 = self.args.get('x3', 2730)
            # 图像分割参数
            # slide_predict = self.args.get('slide_predict', True)
            # slide_size = self.args.get('slide_size', [512, 512])
            # stride = self.args.get('stride', [400, 400])
            # resize_predict = self.args.get('resize_predict', True)
            # resize_scale = self.args.get('resize_scale', 1.0)
            # seg_save_path = self.args.get('seg_save_path')

        # 图像拼接+分割-----------------------------------------------------------------------------------
            if is_auto_seg:
                # try:
                    # 图像拼接
                    start_time = time.time()
                    concat_image_path, concat_matrix = self.auto_concat(self.img1,self.img2,self.img1_path,self.img2_path,concat_save_path,x1,x2,x3)
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    self.signal_process_complete.emit(f"图像拼接完成: {str(concat_matrix.shape)}\n耗时: {elapsed_time:.2f}秒")
                    # 图像分割
                    start_time = time.time()
                    mask_path, mask = self.auto_seg(concat_matrix,concat_image_path,self.args)
                    # mask = self.auto_seg_v2(concat_matrix,self.code,self.args)  # 优化后
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    
                    # # 发送信号，用于后续处理
                    signals.img_postprocess_signal.emit(mask_path, mask)

                    self.signal_process_complete.emit(f"图像分割成功: {str(mask.shape)}\n耗时: {elapsed_time:.2f}秒")
                # except Exception as e:
                #     print(f"图像自动分割失败: {str(e)}")
                #     self.signal_process_complete.emit(f"图像自动分割失败: {str(e)}")  
            else:
                # try:
                    # 遍历拼接后的图像进行分割
                    for root, _ , files in os.walk(img_path):
                        if not self._is_running:  # 检查是否需要停止线程
                            break
                        if files != [] and len(files) == 2:
                            # 记录开始时间
                            start_time = time.time()
                            concat_image_path, concat_matrix=self.one_concat(root,files,concat_save_path,x1,x2,x3)
                            end_time = time.time()
                            elapsed_time = end_time - start_time
                            self.signal_process_complete.emit(f"图像拼接完成: {str(concat_matrix.shape)}\n耗时: {elapsed_time:.2f}秒")
                            # self.signal_concat_complete.emit(concat_matrix)
                            # cv2.imshow("concat", concat_matrix)
                            # cv2.waitKey(0)
                            start_time = time.time()
                            # code =os.path.join(root.split("\\")[-2], root.split("\\")[-1])
                            mask_path, mask = self.auto_seg(concat_matrix,concat_image_path,self.args)
                            end_time = time.time()
                            elapsed_time = end_time - start_time
                            self.signal_process_complete.emit(f"图像分割成功: {str(mask.shape)}\n耗时: {elapsed_time:.2f}秒")

                    self.signal_process_complete.emit("全部图像分割完成")
                    print("全部图像分割完成")
                # except Exception as e:
                #     print(f"图像分割失败: {str(e)}")
                #     self.signal_process_complete.emit(f"图像分割失败: {str(e)}")
                 
        else:  # 非边缘检测
            pass

        self._is_running = False

    def stop(self):
        self._is_running = False
        self.wait()  # 等待线程结束
        # self.quit()  # 退出线程

if __name__ == "__main__":
    app = QApplication(sys.argv)
    thread = ImgSegThread()
    thread.args = {
        'img_path': 'E:\\big_root_sy stem\\data',
        'concat_save_path': 'E:\\big_root_system\\output\\concated',
        'seg_save_path': 'E:\\big_root_system\\output\\seg',
        'model_path': 'E:\\big_root_system\\models\\segment\\bigbox_segformer',
        'device': 'gpu',
        'use_trt': False,
        'use_paddle_trt': False,
        'slide_size': [512, 512],
        'stride': [400, 400]
    }
    thread.start()
    sys.exit(app.exec_())

    # import os
    #
    # output_path = r'E:\big_root_system\output\concated'
    # if not os.access(os.path.dirname(output_path), os.W_OK):
    #     print("No write permission to the directory")