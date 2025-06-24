
# Big Root Box System 项目文档

## 项目简介
本项目是一个基于PyQt5的图形界面应用程序，主要用于植物根系图像处理与分析，集成了图像采集、拼接、分割、去噪、修复等完整工作流，并提供串口通信功能。通过多线程管理实现高效处理，集成FastDeploy框架支持GPU加速和TensorRT优化，满足根系表型分析的专业需求。
#### 1. 采集界面
![](.README_images/根系图像采集界面.png)
#### 2. 处理界面
![](.README_images/根系图像处理界面.png)
## 核心功能
### 1. 图像处理模块
- **图像捕获**：通过`ImgCapture_Thread`线程实现自动化图像采集与保存，支持二维码扫描定位和批量拍摄<mcfile name="img_capture_thread.py" path="e:\big_root_system\threads\img_capture_thread.py"></mcfile>
- **图像拼接与分割**：提供滑动窗口推理(`slide_predict`)和等比例缩放推理(`resize_predict`)两种模式，支持大尺寸图像分割<mcfile name="img_seg_thread.py" path="e:\big_root_system\threads\img_seg_thread.py"></mcfile>
- **根系去噪**：基于深度学习模型的图像去噪处理，去除土壤背景和噪声干扰<mcfile name="img_process_thread.py" path="e:\big_root_system\threads\img_process_thread.py"></mcfile>
- **根系修复**：通过图像修复算法填补根系图像中的缺失部分，优化分析结果
- **表型参数计算**：自动提取根系面积、长度、直径、角度等18项表型参数<mcfile name="img_process_widget.py" path="e:\big_root_system\sub_widgets\img_process_widget.py"></mcfile>

### 2. 模型管理模块
- **多模型支持**：支持分割、去噪、修复等多种模型加载与推理
- **硬件加速**：支持GPU、CPU和TensorRT优化，可根据硬件配置自动选择最佳推理方式<mcfile name="model_load_thread.py" path="e:\big_root_system\threads\model_load_thread.py"></mcfile>
- **动态形状优化**：自动适配不同输入尺寸，优化模型推理性能

### 3. 串口通信模块
- **设备管理**：通过`SerialDevice`类实现串口设备的自动检测、连接与管理<mcfile name="driver_serial.py" path="e:\big_root_system\drivers\driver_serial.py"></mcfile>
- **数据收发**：支持自定义波特率和数据格式，实时监控串口通信状态
- **集成控制**：与图像采集模块联动，实现硬件设备的自动化控制

## 目录结构
```
e:\big_root_system
├── .gitignore               # Git忽略规则
├── main.py                  # 主程序入口，集成各功能模块的主界面
├── config\                  # 配置文件
│   └── default_cfg.py       # 系统默认配置
├── data\                    # 输入数据（Git忽略）
├── drivers\                 # 硬件驱动
│   ├── camera.py            # 相机驱动
│   └── driver_serial.py     # 串口驱动
├── models\                  # 模型文件（Git忽略）
│   ├── inpaint\             # 修复模型
│   └── segment\             # 分割模型
├── output\                  # 输出结果（Git忽略）
├── signals\                 # 全局信号定义
│   ├── global_signals.py    # 自定义信号
│   └── global_vars.py       # 全局变量
├── sub_widgets\             # 界面组件
│   ├── img_capture_widget.py# 图像捕获界面
│   ├── img_process_widget.py# 图像处理主界面
│   ├── serial_assist_widget.py# 串口助手界面
│   ├── serial_setting_dialog.py # 串口设置对话框
│   └── widgets_v1\          # 历史版本界面组件
├── threads\                 # 后台线程
│   ├── img_capture_thread.py# 图像捕获线程
│   ├── img_process_thread.py# 图像处理线程
│   ├── img_seg_thread.py    # 图像分割线程
│   ├── model_load_thread.py # 模型加载线程
│   └── utils\               # 线程工具函数
├── tools\                   # 通用工具
│   └── utils.py             # 字节解码等工具函数
├── ui\                      # 界面设计文件
│   ├── *.ui                 # Qt Designer界面文件
│   └── resource.qrc         # 资源文件
└── requirements.txt         # 环境依赖
```

## 环境配置
### 1. 基础依赖安装
```bash
# 创建虚拟环境（推荐）
conda create -n root_analysis python=3.9
conda activate root_analysis

# 安装依赖（项目根目录下）
pip install -r requirements.txt

# 可选：根据硬件配置选择GPU或CPU版本
# GPU环境（推荐）
pip install pyqt5 pyserial opencv-python numpy fastdeploy-gpu onnxruntime-gpu

# CPU环境
# pip install pyqt5 pyserial opencv-python numpy fastdeploy onnxruntime
```

### 2. 硬件配置（可选）
- **相机驱动**：确保相机驱动已正确安装并能被OpenCV识别
- **串口权限**（Linux/macOS）：
  ```bash
  sudo usermod -aG dialout $USER
  reboot
  ```

## 使用指南
### 1. 快速启动
```bash
# 项目根目录下运行
python main.py
```
主界面将显示所有可用功能模块的选项卡，包括图像采集、图像处理和串口助手。

### 2. 图像分割流程
1. 在主界面选择"根系图像处理"选项卡
2. 点击"模型路径"按钮，选择包含以下文件的模型目录：
   - model.pdmodel（模型结构）
   - model.pdiparams（模型参数）
   - deploy.yaml（配置文件）
3. 点击"加载模型"，选择加速方式（GPU/CPU/TensorRT）
4. 设置分割参数：
   - 滑动窗口大小（默认512x512）
   - 滑动步长（默认256）
   - 推理模式（滑动窗口/等比例缩放）
5. 点击"开始分割"，处理进度将显示在日志窗口

### 3. 批量图像处理
1. 在"根系图像处理"界面选择"批量处理"选项卡
2. 设置输入目录（包含待处理图像）和输出目录
3. 勾选需要执行的处理步骤（分割、去噪、修复、表型分析）
4. 点击"开始批量处理"，系统将自动处理所有图像并保存结果

## 模型下载
| 模型类型 | 版本 | 下载链接 | 大小 |
|---------|------|---------|------|
| 根系分割模型 | v1.0.0 | [GitHub Releases](https://github.com/Tansong666/big_root_box_system/releases/download/v1.0.0/bigbox_segformer.zip) | 145MB |
| 根系去噪模型 | v1.0.0 | [GitHub Releases](https://github.com/Tansong666/big_root_box_system/releases/download/v1.0.0/denoise_model.zip) | 98MB |
| 根系修复模型 | v1.0.0 | [GitHub Releases](https://github.com/Tansong666/big_root_box_system/releases/download/v1.0.0/inpaint_model.zip) | 210MB |
| 测试数据集 | v1.0 | [GitHub Releases](https://github.com/Tansong666/big_root_box_system/releases/download/v1.0/root_data_demo.zip) | 380MB |

## 注意事项
1. **性能优化**：
   - 大尺寸图像推荐使用滑动窗口模式
   - TensorRT加速首次运行会生成缓存文件（约1-2分钟），后续运行将显著加快
   - 批量处理时建议设置合理的线程数，避免内存溢出

2. **文件管理**：
   - 图像数据建议存放在`data/`目录下
   - 模型文件建议存放在`models/`目录下
   - 所有输出结果将自动保存到`output/`目录，按日期和处理类型分类

3. **常见问题**：
   - 模型加载失败：检查模型文件是否完整，配置是否正确
   - 图像显示异常：尝试调整图像缩放比例或重启程序
   - 串口连接失败：检查设备是否正确连接，权限是否配置

## 开发说明
项目采用模块化设计，主要分为以下几个部分：
- **界面层**：基于PyQt5的图形界面，位于`sub_widgets/`目录
- **业务逻辑层**：后台处理线程，位于`threads/`目录
- **驱动层**：硬件设备接口，位于`drivers/`目录
- **工具层**：通用功能函数，位于`tools/`目录

如需扩展功能，建议通过添加新的线程类和界面组件实现，保持现有架构的一致性。
        