import numpy as np
import cv2
import os

# def concat(path_img,path_save):
#     # 单次图片的拼接
#     pass


def pinjie(path_img,path_save):
    # final_matrix = None
    for root, dirs, files in os.walk(path_img): # os.walk() 它会遍历 path_img 指定的路径下的所有子目录和文件。
        if files != []:
            img1_path=os.path.join(root,files[0]) 
            img2_path=os.path.join(root,files[1]) 
            save_path = os.path.join(path_save, root.split("\\")[-2], root.split("\\")[-1] + '.png')
            if os.path.exists(save_path): # 判断文件是否存在
                pass
            else:
                print(img1_path)
                print(img2_path)
                img2 = cv2.imread(img1_path)  # 读取图片       img2 = cv2.imread(path2)
                img1 = cv2.imread(img2_path)
                # img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)
                # img2 = cv2.rotate(img2, cv2.ROTATE_90_COUNTERCLOCKWISE)
                # the image height
                sum_rows = img1.shape[0]
                # the image length
                sum_cols = img1.shape[1]
                # 对F0图片进行剪切
                x1 = 2000  # 第一刀
                x2 = 2680  # 第二刀
                x3 = 3350  # 第三刀
                part1 = img1[0:sum_rows, 0:x1]  
                part2 = img1[0:sum_rows, x1:x2]
                part3 = img1[0:sum_rows, x2:x3]
                part4 = img1[0:sum_rows, x3:sum_cols]
                # 对F1图片进行剪切
                part5 = img2[0:sum_rows, 0:x1]
                part6 = img2[0:sum_rows, x1:x2]
                part7 = img2[0:sum_rows, x2:x3]
                part8 = img2[0:sum_rows, x3:sum_cols]
                # 设定一个大小和原图一样的零矩阵
                '''
                通过将数组初始化为全零，我们可以确保final_matrix中的所有像素值都是黑色。这个数组可以用来存储拼接后的图片数据，每个像素由三个通道（红、绿、蓝）组成。
                这行代码创建了一个名为final_matrix的变量，并使用np.zeros()函数初始化了一个三维的NumPy数组。
                这个数组的形状是(sum_rows, sum_cols, 3)，其中sum_rows和sum_cols是之前定义的变量，表示拼接后图片的总行数和总列数。
                数组的数据类型被指定为np.uint8，表示每个元素是一个8位无符号整数。
                '''
                final_matrix = np.zeros((sum_rows, sum_cols, 3), np.uint8)
                  
                # 将需要的部分拼接起来
                # final_matrix[0:sum_rows, 0:x1] = part5
                # final_matrix[0:sum_rows, x1:x2] = part2
                # final_matrix[0:sum_rows, x2:x3] = part7
                # final_matrix[0:sum_rows, x3:sum_cols] = part4
                final_matrix[0:sum_rows, 0:x1] = part1
                final_matrix[0:sum_rows, x1:x2] = part6
                final_matrix[0:sum_rows, x2:x3] = part3
                final_matrix[0:sum_rows, x3:sum_cols] = part8
                # 将拼接好的图片保存为F2
                # count = root[-9:-5]
                # save_dirs = os.path.join(path_save,root.split("\\")[-2])
                # if not os.path.isdir(save_dirs):
                #     os.makedirs(save_dirs)
                print(save_path)
                '''
                cv2.imwrite()函数是OpenCV库中用于将图像保存到文件的函数之一。
                它接受三个参数：保存路径、图像矩阵和可选的保存参数。在这个例子中，save_path1是保存路径，final_matrix是要保存的图像矩阵。
                第三个参数是一个可选的保存参数，用于指定保存图像的质量。
                在这个例子中，使用了[int(cv2.IMWRITE_JPEG_QUALITY), 100]作为保存参数。
                这个参数告诉cv2.imwrite()函数将图像保存为JPEG格式，并将保存的图像质量设置为100。JPEG质量的范围是0到100，其中100表示最高质量。
                '''
                cv2.imwrite(save_path, final_matrix)
        else:
            pass
    # return final_matrix

if __name__ == '__main__':
    path_img=r'G:\Root_data\captured'
    path_save=r'G:\Root_data\concated\captured'
    a=pinjie(path_img,path_save)

    # 创建文件夹
    if not os.path.exists(path_save):
        os.makedirs(path_save)