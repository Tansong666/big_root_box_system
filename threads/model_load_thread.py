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
        self.is_seg_model = True
        self.is_inpaint_model = None
 
    def build_option(self):
        option = fd.RuntimeOption()

        if self.device.lower() == "gpu":
            option.use_gpu(0)

        # 添加TRT缓存验证逻辑
        # if self.use_trt and os.path.exists('paddle_seg.trt'):
        #     option.use_trt_backend()
        #     # option.enable_trt_fp16()
        #     option.set_trt_cache_file('paddle_seg.trt')
        #     return option
        if self.use_trt and os.path.exists('trt_cache/segformerb2_fp16.trt'):
            option.use_trt_backend()
            option.enable_trt_fp16()
            option.set_trt_cache_file('trt_cache/segformerb2_fp16.trt')
            return option

        if self.use_trt:
            option.use_trt_backend()
            option.set_trt_input_shape("x", [1, 3, 256, 256], [1, 3, 512, 512],
                                       [1, 3, 1024, 1024])
            # option.enable_trt_fp16()
            option.set_trt_cache_file('segformerb2.trt')
            return option
            # option.trt_opt_shape_shrink = True  # 启用动态shape优化
        elif self.use_paddle_trt:
            option.use_trt_backend()
            option.enable_paddle_to_trt()
            option.enable_paddle_trt_collect_shape()
            option.set_trt_input_shape("x", [1, 3, 256, 256], [1, 3, 512, 512],
                                       [1, 3, 1024, 1024])
        return option

    def seg_model(self):
        try:
            runtime_option = self.build_option()
            model_file = os.path.join(self.model_path, "model.pdmodel")
            params_file = os.path.join(self.model_path, "model.pdiparams")
            config_file = os.path.join(self.model_path, "deploy.yaml")
            model = fd.vision.segmentation.PaddleSegModel(
                model_file, params_file, config_file, runtime_option=runtime_option)
            self.model_loaded.emit(model)  # 发送模型加载完成信号
        except Exception as e:
            self.load_error.emit(str(e))  # 发送模型加载错误信号

    def inpaint_model(self):
        try:
            option = fd.RuntimeOption()
            option.set_model_path(r"E:\big_root_system\models\inpaint\EUGAN.onnx", model_format=ModelFormat.ONNX)
            # **** GPU 配置 ***
            option.use_gpu(0)
            option.use_trt_backend()
            # option.set_trt_cache_file('EUGAN.trt')
            option.trt_option.serialize_file = 'trt_cache/EUGAN.trt'

            # 初始化构造runtime
            runtime = fd.Runtime(option)
            # # 获取模型输入名
            # input_name = runtime.get_input_info(0).name
            # print(input_name)  # x 为模型输入名
            # # 构造随机数据进行推理
            # results = runtime.infer({
            #     input_name: np.random.rand(1, 1, 640, 384).astype("float32")
            # })
            # print(results[0].shape)
            self.model_loaded.emit(runtime)  # 发送模型加载完成信号
        except Exception as e:
            self.load_error.emit(str(e))  # 发送模型加载错误信号
        # return runtime

    def run(self):
        if self.is_seg_model:
            self.seg_model()
        if self.is_inpaint_model:
            self.inpaint_model()
        

if __name__ == "__main__":
    import fastdeploy as fd
    from fastdeploy import ModelFormat
    import numpy as np

# 下载模型并解压
# model_url = "https://bj.bcebos.com/fastdeploy/models/mobilenetv2.onnx"
# fd.download(model_url, path=".")

    option = fd.RuntimeOption()

    option.set_model_path(r"E:\big_root_system\models\inpaint\EUGAN.onnx", model_format=ModelFormat.ONNX)

    # **** GPU 配置 ***
    option.use_gpu(0)
    option.use_trt_backend()
    option.set_trt_cache_file('EUGAN.trt')
    option.trt_option.serialize_file = 'EUGAN.trt'

    # 初始化构造runtime
    runtime = fd.Runtime(option)

    # 获取模型输入名
    input_name = runtime.get_input_info(0).name
    print(input_name)  # x 为模型输入名

    # 构造随机数据进行推理
    results = runtime.infer({
        input_name: np.random.rand(1, 1, 640, 384).astype("float32")
    })

    print(results[0].shape)