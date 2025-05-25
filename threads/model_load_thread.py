from turtle import update
from PyQt5.QtCore import QThread, pyqtSignal
import fastdeploy as fd
import os

class ModelLoadThread(QThread):
    model_loaded = pyqtSignal(object)  # 定义信号，用于通知模型加载完成
    load_error = pyqtSignal(str)  # 定义信号，用于通知模型加载错误
    # record = pyqtSignal(float, float)

    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path
        self.device = 'gpu'
        self.use_trt = False
        self.use_paddle_trt = False

    def build_option(self):
        option = fd.RuntimeOption()

        if self.device.lower() == "gpu":
            option.use_gpu(0)

        # 添加TRT缓存验证逻辑
        if self.use_trt and os.path.exists('paddle_seg.trt'):
            option.use_trt_backend()
            option.set_trt_cache_file('paddle_seg.trt')
            return option

        if self.use_trt:
            option.use_trt_backend()
            option.set_trt_input_shape("x", [1, 3, 256, 256], [1, 3, 512, 512],
                                       [1, 3, 1024, 1024])
            option.set_trt_cache_file('paddle_seg.trt')
            return option
            # option.trt_opt_shape_shrink = True  # 启用动态shape优化
        if self.use_paddle_trt:
            option.use_trt_backend()
            option.enable_paddle_to_trt()
            option.enable_paddle_trt_collect_shape()
            option.set_trt_input_shape("x", [1, 3, 256, 256], [1, 3, 1024, 1024],
                                       [1, 3, 2048, 2048])
        return option

    def run(self):
        # import time
        try:
            # start_time = time.time()
            runtime_option = self.build_option()
            model_file = os.path.join(self.model_path, "model.pdmodel")
            params_file = os.path.join(self.model_path, "model.pdiparams")
            config_file = os.path.join(self.model_path, "deploy.yaml")
            model = fd.vision.segmentation.PaddleSegModel(
                model_file, params_file, config_file, runtime_option=runtime_option)
            # end_time = time.time()
            # self.record.emit(start_time, end_time)
            self.model_loaded.emit(model)  # 发送模型加载完成信号
        except Exception as e:
            self.load_error.emit(str(e))  # 发送模型加载错误信号

    # def run(self):  # 优化后的模型加载函数 并行加载模型文件
    #     import time
    #     try:
    #         # 并行加载模型文件
    #         from concurrent.futures import ThreadPoolExecutor
    #         start_time = time.time()
    #         with ThreadPoolExecutor() as executor:
    #             model_future = executor.submit(os.path.join, self.model_path, "model.pdmodel")
    #             params_future = executor.submit(os.path.join, self.model_path, "model.pdiparams")
    #             config_future = executor.submit(os.path.join, self.model_path, "deploy.yaml")
                
    #             model_file = model_future.result()
    #             params_file = params_future.result()
    #             config_file = config_future.result()

    #         runtime_option = self.build_option()
    #         # 直接使用文件路径加载模型
    #         model = fd.vision.segmentation.PaddleSegModel(
    #             model_file, params_file, config_file, runtime_option=runtime_option)
    #         end_time = time.time()
    #         self.record.emit(start_time, end_time)
    #         # print(f"模型加载耗时: {end_time - start_time:.2f} 秒")
    #         # print('------------------------------------------------------------')
    #         # print(f"模型加载耗时: {end_time - start_time:.2f} 秒")

    #         self.model_loaded.emit(model)
    #     except Exception as e:
    #         self.load_error.emit(str(e))