
# Big Root Box System 项目文档

## 项目简介
本项目是一个基于PyQt5的图形界面应用程序，主要用于图像处理（图像采集、拼接、分割）与串口通信辅助功能。支持通过线程管理耗时操作（如图像分割、模型加载），并集成了[**FastDeploy**](https://github.com/PaddlePaddle/FastDeploy)模型推理框架实现高效的图像分割任务。

## 核心功能
1. **图像处理**  
   - 图像捕获：通过`ImgCapture_Thread`线程实现图像采集与保存（参考`<mcfile name="img_capture_widget.py" path="e:\big_root_system\tasks\img_capture_widget.py"></mcfile>`）。  
   - 图像拼接与分割：支持滑动窗口推理（`slide_predict`）和等比例缩放推理（`resize_predict`）两种模式，通过`img_seg_thread.py`和`img_seg_thread_v2.py`线程实现（参考`<mcfile name="img_seg_thread.py" path="e:\big_root_system\threads\img_seg_thread.py"></mcfile>`）。  
   - 模型加载：通过`ModelLoadThread`线程加载PaddleSeg模型，支持GPU加速和TensorRT优化（参考`<mcfile name="model_load_thread.py" path="e:\big_root_system\threads\model_load_thread.py"></mcfile>`）。

2. **串口通信**  
   - 串口设备管理：通过`SerialDevice`类实现串口的打开、关闭、读写操作（参考`<mcfile name="driver_serial.py" path="e:\big_root_system\drivers\driver_serial.py"></mcfile>`）。  
   - 串口助手界面：提供串口选择、数据收发功能（参考`<mcfile name="serial_assist_widget.py" path="e:\big_root_system\tasks\serial_assist_widget.py"></mcfile>`）。

## 目录结构
```
e:\big_root_system
├── .gitignore               # Git忽略规则
├── main.py                  # 主程序入口（待补充）
├── config\                  # 配置文件
├── data\                    # 输入数据（Git忽略）
├── drivers\                 # 硬件驱动
│   ├── camera.py            # 相机驱动
│   └── driver_serial.py     # 串口驱动
├── models\                  # 模型文件（Git忽略）
├── output\                  # 输出结果（Git忽略）
├── signals\                 # 全局信号定义
├── tasks\                   # 界面组件
│   ├── img_capture_widget.py# 图像捕获界面
│   ├── img_seg_widget.py    # 图像分割界面
│   └── serial_assist_widget.py# 串口助手界面
│   ├── img_denoise_widget.py  # 根系去噪界面（todo 待做）
│   └── img_analysis_widget.py # 根系修复界面（todo 待做）
├── threads\                 # 后台线程
│   ├── img_seg_thread.py    # 图像分割线程
│   ├── model_load_thread.py # 模型加载线程
│   └── utils\               # 线程工具函数
├── tools\                   # 通用工具
│   └── utils.py             # 字节解码等工具函数
└── ui\                      # 界面设计文件
│   ├── *.ui                 # Qt Designer界面文件
│   └── resource.qrc         # 资源文件
├── requirements.txt         # 环境依赖

```

## 依赖安装
1. 基础依赖：  
   ```bash
   # 创建虚拟环境（可选）
   conda create -n pyqt5 python=3.9
   conda activate pyqt5
   # 安装依赖(切换到项目根目录运行命令)
   pip install -r requirements.txt
   # pip install pyqt5 pyserial opencv-python numpy fastdeploy-gpu  # GPU环境
   # 或 pip install pyqt5 pyserial opencv-python numpy fastdeploy  # CPU环境
   ```
2. 串口权限（Linux/macOS）：  
   参考`<mcfile name="cmd.txt" path="e:\big_root_system\tools\cmd.txt"></mcfile>`中的命令为用户添加串口权限：
   ```bash
   sudo usermod -aG dialout <your_username>
   reboot
   ```

## 使用说明
1. **启动程序**：运行`main.py`启动主界面。
2. **启动子程序**：运行`tasks/img_capture_widget.py`启动图像捕获界面，运行`tasks/img_seg_widget.py`启动图像分割界面，运行`tasks/serial_assist_widget.py`启动串口助手界面。
3. **图像分割流程**：  
   - 在图像分割界面（`img_seg_widget.py`）中选择模型路径（`model.pdmodel`/`model.pdiparams`）。  
   - 点击"加载模型"，通过`ModelLoadThread`线程加载模型（支持GPU/CPU/TensorRT加速）。  
   - 选择输入图像路径，配置分割参数（滑动窗口大小、步长等），启动分割任务。  
4. **串口通信（调试）**：  
   - 在串口助手界面（`serial_assist_widget.py`）中选择串口、波特率，点击"打开串口"。  
   - 输入数据后点击"发送"，或通过"接收"按钮读取串口数据。

## 配置说明
- 模型路径：需包含`model.pdmodel`（模型结构）、`model.pdiparams`（模型参数）和`deploy.yaml`（配置文件）。  
- 图像分割参数：可通过界面调整`slide_predict`（滑动窗口开关）、`slide_size`（窗口大小）、`stride`（步长）等参数（参考`<mcfile name="img_seg_thread.py" path="e:\big_root_system\threads\img_seg_thread.py"></mcfile>`）。  

## 模型下载地址
- 根系分割模型：[GitHub Releases 下载](https://github.com/Tansong666/big_root_box_system/releases/download/v1.0.0/bigbox_segformer.zip)
- 测试数据：[GitHub Releases 下载](https://github.com/Tansong666/big_root_box_system/releases/download/v1.0/root_data_demo.zip)  
- 去噪模型（待发布）：[待上传，后续更新链接]


## 注意事项
- 图像和模型文件建议存储在`data/`和`models/`目录下（已被`.gitignore`忽略，避免提交大文件）。  
- 长时间任务（如图像分割）会通过线程异步执行，界面不会卡顿（参考`<mcsymbol name="run" filename="img_seg_thread.py" path="e:\big_root_system\threads\img_seg_thread.py" startline="200" type="function"></mcsymbol>`方法）。

        
