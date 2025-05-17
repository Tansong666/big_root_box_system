import os
import cv2
# from PIL import Image

# def get_image_info(image_path):
#     """
#     获取图像的常用信息。

#     :param image_path: 图像文件的路径
#     :return: 包含图像常用信息的字典
#     """
#     if not os.path.exists(image_path):
#         print(f"文件 {image_path} 不存在。")
#         return None

#     try:
#         with Image.open(image_path) as img:
#             info = {
#                 "文件路径": image_path,
#                 "文件大小": os.path.getsize(image_path),
#                 "图像格式": img.format,
#                 "图像模式": img.mode,
#                 "图像尺寸": img.size,
#                 "宽度": img.width,
#                 "高度": img.height
#             }
#             return info
#     except Exception as e:
#         print(f"读取图像 {image_path} 时出错: {e}")
#         return None

def get_image_info(image_path):
    """
    获取图像的常用信息。

    :param image_path: 图像文件的路径
    :return: 包含图像常用信息的字典
    """
    if not os.path.exists(image_path):
        print(f"文件 {image_path} 不存在。")
        return None

    try:
        img = cv2.imread(image_path,0)
        if img is None:
            print(f"无法读取图像 {image_path}。")
            return None
        height, width, channels = img.shape if len(img.shape) == 3 else (img.shape[0], img.shape[1], 1)
        info = {
            "文件路径": image_path,
            "文件大小": os.path.getsize(image_path),
            "图像高度": height,
            "图像宽度": width,
            "图像通道数": channels
        }
        return info
    except Exception as e:
        print(f"读取图像 {image_path} 时出错: {e}")
        return None


if __name__ == "__main__":
    image_path = r"E:\big_root_system\output\0510\0510\171.png"  # 替换为你的图像路径
    image_info = get_image_info(image_path)
    if image_info:
        print("图像信息:")
        for key, value in image_info.items():
            print(f"{key}: {value}")

    # import cv2
    # import numpy as np

    # # 创建单通道二维数组（灰度图）
    # img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

    # # 保存为不同格式
    # cv2.imwrite("gray.jpg", img)  # 可能变为三通道
    # cv2.imwrite("gray.png", img)  # 保持单通道

    # # 默认读取为三通道（BGR）
    # jpg_img = cv2.imread("gray.jpg")  
    # print(jpg_img.shape)  # 输出 (100, 100, 3)

    # # 显式读取为单通道
    # png_img = cv2.imread("gray.png", cv2.IMREAD_GRAYSCALE)  
    # print(png_img.shape)  # 输出 (100, 100)