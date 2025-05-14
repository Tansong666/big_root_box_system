import fastdeploy as fd
import cv2
import os
import numpy as np
# from fastdeploy.libs.fastdeploy_main.vision import SegmentationResult

# python infer.py --model ./PP_LiteSeg_B_STDC2_cityscapes_without_argmax_infer --image ./cityscapes_demo.png --device gpu
# python threads/utils/seg.py --model models/segment/libox_segformer --image data/2_2_3_5.png --device gpu

def parse_arguments():
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, help="Path of PaddleSeg model.")
    parser.add_argument(
        "--image", type=str, required=True, help="Path of test image file.")
    parser.add_argument(
        "--device",
        type=str,
        default='gpu',
        help="Type of inference device, support 'kunlunxin', 'cpu' or 'gpu'.")
    parser.add_argument(
        "--use_trt",
        type=ast.literal_eval,
        default=False,
        help="Wether to use tensorrt.")
    parser.add_argument(
        "--use_paddle_trt",
        type=ast.literal_eval,
        default=False,
        help="Wether to use paddle tensorrt.")
    parser.add_argument(
        "--window_size",
        type=int,
        nargs=2,
        default=[1024, 1024],
        help="Size of the sliding window, e.g., --window_size 512 512")
    parser.add_argument(
        "--stride",
        type=int,
        nargs=2,
        default=[800, 800],
        help="Stride of the sliding window, e.g., --stride 256 256")
    return parser.parse_args()


#多行注释 
'''
option = fd.RuntimeOption()
# 切换不同设备
option.use_kunlunxin()
option.use_cpu()
option.use_gpu()

# 切换不同后端
option.use_paddle_backend() # Paddle Inference
option.use_trt_backend() # TensorRT
option.use_openvino_backend() # OpenVINO
option.use_ort_backend() # ONNX Runtime
'''

def build_option(args):
    option = fd.RuntimeOption()

    if args.device.lower() == "gpu":
        option.use_gpu()

    if args.use_trt:
        option.use_trt_backend()
        option.set_trt_input_shape("x", [1, 3, 256, 256], [1, 3, 1024, 1024],
                                   [1, 3, 2048, 2048])
        option.set_trt_cache_file('paddle_seg.trt')
    if args.use_paddle_trt:
        option.use_trt_backend()
        option.enable_paddle_to_trt()
        option.enable_paddle_trt_collect_shape()
        option.set_trt_input_shape("x", [1, 3, 256, 256], [1, 3, 1024, 1024],
                                   [1, 3, 2048, 2048])
    return option

if __name__ == "__main__":
    args = parse_arguments()
    # 定义滑动窗口预测的标志
    slide_predict = False
    resize_predict = True

    # settting for runtime
    runtime_option = build_option(args)
    model_file = os.path.join(args.model, "model.pdmodel")
    params_file = os.path.join(args.model, "model.pdiparams")
    config_file = os.path.join(args.model, "deploy.yaml")
    model = fd.vision.segmentation.PaddleSegModel(
        model_file, params_file, config_file, runtime_option=runtime_option)

    # Load the image
    im = cv2.imread(args.image)

    if slide_predict:
        h, w = im.shape[:2]
        window_h, window_w = args.window_size
        stride_h, stride_w = args.stride

        # Initialize the result label map
        result_label_map = np.zeros((h, w), dtype=np.uint8)

        # Sliding window inference
        for y in range(0, h - window_h + 1, stride_h):
            for x in range(0, w - window_w + 1, stride_w):
                window = im[y:y + window_h, x:x + window_w]
                # # 显示window以图片的形式
                # cv2.imshow("window", window)
                # cv2.waitKey(0)
                # 保存window为图片
                # cv2.imwrite(f"output/window/window_{y}_{y + window_h}_{x}_{x + window_w}.png", window)

                # # 将window转换为RGB格式
                # window = cv2.cvtColor(window, cv2.COLOR_BGR2RGB)
                # cv2.imshow("window_rgb", window)
                # cv2.waitKey(0)
                # 将window转化为BGR格式
                window = cv2.cvtColor(window, cv2.COLOR_RGB2BGR)
                # cv2.imshow("window_bgr", window)
                # cv2.waitKey(0)

                window_result = model.predict(window)
                window_label_map = np.array(window_result.label_map).reshape(window_result.shape)
                # # 调试
                # window_label_map_uint8 = window_label_map.astype(np.uint8)
                # # 将大于0的值设置为255
                # window_label_map_uint8[window_label_map_uint8 > 0] = 255
                # # 显示window_label_map以图片的形式
                # cv2.imshow("window_label_map", window_label_map_uint8)
                # cv2.waitKey(0)
                # # 保存window_label_map为图片
                # cv2.imwrite(f"output/mask/window_label_map_{y}_{y + window_h}_{x}_{x + window_w}.png", window_label_map_uint8)

                result_label_map[y:y + window_h, x:x + window_w] = window_label_map

        mask = np.where(result_label_map > 0, 255, 0).astype(np.uint8)

    else:
        if resize_predict:
            # 对图像进行等比例缩放
            scale = 0.5  # 缩放比例
            im = cv2.resize(im, None, fx=scale, fy=scale)
        # 直接进行推理
        result = model.predict(im)
        label_map = np.array(result.label_map).reshape(result.shape)
        mask = np.where(label_map > 0, 255, 0).astype(np.uint8)
        # 将mask恢复到原始图像大小
        if resize_predict:
            # mask = cv2.resize(mask, (int(im.shape[1] / scale), int(im.shape[0] / scale)))  # 默认采用双线性插值法
            # 插值法
            mask = cv2.resize(mask, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)  # interpolation是插值法

    # # 显示mask以图片的形式
    # cv2.imshow("mask", mask)
    # cv2.waitKey(0)

    # 保存掩码图
    cv2.imwrite("output/segmentation_mask-resize-recover.png", mask)
    print("Segmentation mask saved as segmentation_mask.png")

    # # # visualize
    # # vis_im = fd.vision.vis_segmentation(window, window_result, weight=0.5)
    # # cv2.imwrite("vis_img_test.png", vis_im)