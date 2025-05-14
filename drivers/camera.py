import numpy
from pypylon import pylon 
import cv2
from datetime import date, datetime
import os
# search device and get device
import serial

def search_get_device(): 
    tl_factory = pylon.TlFactory.GetInstance()  # 创建一个TlFactory实例
    dev_info1 = tl_factory.EnumerateDevices()[0] # 获取设备信息
    camera1 = pylon.InstantCamera(tl_factory.CreateDevice(dev_info1)) # 创建相机实例
    return camera1  # 返回相机实例


def save_multi_image():
    cam = search_get_device()
    img = pylon.PylonImage()
    num_img_to_save = 5
    cam.Open()
    cam.StartGrabbing()  # Starts the grabbing for a maximum number of images.
    for i in range(num_img_to_save):
        with cam.RetrieveResult(2000) as result:
            # Calling AttachGrabResultBuffer creates another reference to the
            # grab result buffer. This prevents the buffer's reuse for grabbing.
            img.AttachGrabResultBuffer(result)
            # print("Img reference:",img)
            # print("Result reference",result)
            # The JPEG format that is used here supports adjusting the image
            # quality (100 -> best quality, 0 -> poor quality).
            ipo = pylon.ImagePersistenceOptions()
            quality = 100 - i * 20
            # quality = 100
            ipo.SetQuality(quality)
            filename = f"saved_pypylon_img_{quality}.jpeg"
            img.Save(pylon.ImageFileFormat_Jpeg, filename)#, ipo)
            img.Release()
    cam.StopGrabbing()
    cam.Close()

def grab_show_image(): 
    cam = search_get_device() # get device
    cam.Open() # 打开相机
    # Grabing Continusely (video) with minimal delay
    cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) # 开始抓取图像
    converter = pylon.ImageFormatConverter() # 转换图像格式
    # converting to opencv bgr format
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed # opencv bgr格式 输出像素格式
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned # 输出位对齐
    # Wait for an image and then retrieve it. A timeout of 5000 ms is used.
    grabResult = cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException) # 等待图像并检索图像
    # Image grabbed successfully?
    # print(dir(grabResult))
    if grabResult.GrabSucceeded(): # 抓取成功
        # Access the image data.
        print("SizeX: ", grabResult.Width) 
        print("SizeY: ", grabResult.Height)
        # img type class 'numpy.ndarray', shape 1944*2592*2
        img = grabResult.Array # 获取图像数据

        print("Gray value of first pixel: ", img[0, 0]) # 获取第一个像素点的灰度值

        # After convert to image(ndarray) shape 1944*2592*3
        image = converter.Convert(grabResult) # 转换图像格式
        weld_img = image.GetArray() # 获取图像数据
        # cv2.namedWindow('test', cv2.WINDOW_NORMAL)
        # cv2.imshow('test', weld_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        print('weld_img_type',type(weld_img)) 
        print('img_type', type(img))
        print(weld_img[weld_img[:,:,1] != img[:,:,0]]) # 比较两个图像的像素值
    else:
        print("Error: ", grabResult.ErrorCode, grabResult.ErrorDescription) # 抓取失败
    grabResult.Release() # 释放图像
    cam.Close() # 关闭相机
# grab_show_image()


def grab_image_save():   # save image  
    cam = search_get_device()
    cam.Open() # 打开相机
    save_img = pylon.PylonImage() # 创建一个PylonImage实例
    # Grabing Continusely (video) with minimal delay
    cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) # 开始抓取图像

    # Wait for an image and then retrieve it. A timeout of 5000 ms is used.
    grabResult = cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException) # 等待图像并检索图像
    # Image grabbed successfully
    # print(dir(grabResult))
    if grabResult.GrabSucceeded(): # 抓取成功
        # save image
        # filename = f"D:/waterroot/{name}.png"
        save_img.AttachGrabResultBuffer(grabResult) # 附加抓取结果缓冲区
        ipo = pylon.ImagePersistenceOptions() # 创建一个ImagePersistenceOptions实例
        ipo.SetQuality(quality=100) # 设置图像质量
        # print(save_img.shape)
        # save_img = numpy.rot90(save_img, k=1 , axes=(0,1))
        # save_img.Save(pylon.ImageFileFormat_Jpeg, filename, ipo)
        # save_img.Release()
        # Access the image data.
        # img type class 'numpy.ndarray', shape 1944*2592*2
        converter = pylon.ImageFormatConverter() # 创建一个ImageFormatConverter实例
        # converting to opencv bgr format
        converter.OutputPixelFormat = pylon.PixelType_BGR8packed # opencv bgr格式 输出像素格式
        converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned # 输出位对齐
        image = converter.Convert(grabResult)  # 转换图像格式
        weld_img = image.GetArray() # 获取图像数据
        # weld_img = img.GetArray()

        # print('img_type', type(weld_img))
        # print(weld_img.shape)
    else:
        print("Error: ", grabResult.ErrorCode, grabResult.ErrorDescription) # 抓取失败
    grabResult.Release()    # 释放图像
    cam.Close() # 关闭相机


    return weld_img

def grab_2image_save(cam,name):
    cam.Open()
    save_img = pylon.PylonImage()
    # Grabing Continusely (video) with minimal delay
    cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    # Wait for an image and then retrieve it. A timeout of 5000 ms is used.
    grabResult = cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    # Image grabbed successfully
    # print(dir(grabResult))
    if grabResult.GrabSucceeded():
        # save image
        filename = f"D:/waterroot/{name}.png"
        save_img.AttachGrabResultBuffer(grabResult)
        ipo = pylon.ImagePersistenceOptions()
        ipo.SetQuality(quality=100)
        # print(save_img.shape)
        # save_img = numpy.rot90(save_img, k=1 , axes=(0,1))
        save_img.Save(pylon.ImageFileFormat_Jpeg, filename, ipo)
        # save_img.Release()
        # Access the image data.
        # img type class 'numpy.ndarray', shape 1944*2592*2
        converter = pylon.ImageFormatConverter()
        # converting to opencv bgr format
        converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        image = converter.Convert(grabResult)
        weld_img = image.GetArray()
        # weld_img = img.GetArray()

        print('img_type', type(weld_img))
        print(weld_img.shape)
    else:
        print("Error: ", grabResult.ErrorCode, grabResult.ErrorDescription)
    grabResult.Release()
    cam.Close()


    return weld_img


# 定义某个外置摄像头拍取图片的函数，并保存图片到指定路径
def capture_qr_image(camera_id, save_path):
    # 初始化摄像头
    cap = cv2.VideoCapture(camera_id)
    # 读取图像
    ret, frame = cap.read()
    # 如果save_path不存在，则创建
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # 获取当前时间作为图像名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_name = f"{save_path}/{timestamp}.jpg"
    # 保存图像
    img = cv2.imwrite(image_name, frame)
    print(f"Image saved: {image_name}")
    # 释放摄像头
    cap.release()
    # return img