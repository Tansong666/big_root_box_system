# 必选参数edge和serving二选一
# is_Edge=False
is_Serving=False

# 当前工作目录的output目录设为根目录
root_path = r'E:\big_root_system'

# concat
is_concat = True
data_path =r'data'
concat_x1 = 2000
concat_x2 = 2700
concat_x3 = 3350
concat_savepath = r'output\concated'

# seg
is_seg = True
seg_runtime_list = ['paddle_inference','onnxruntime','tensorrt','paddle_serving']
seg_model_list = ['segformer','deeplabv3','hardnet','unet']
is_slide=True
crop_size = 1024
stride = 1024
is_resize = True
resize_scale = 1.0
# seg_path
seg_weightpath = r'models\segment\bigbox_segformer'
seg_imagepath = r'output\concated'
seg_savepath = r'output\segmented'


#### 后处理
# 图像去噪
is_denoise = True
post_inputpath = r'output\test\0510'
is_rsa = False
dilate_iters = 10
threshold_area = 1000
is_rba = False
rba_left = 0
rba_right = 0
rba_top = 0
rba_bottom = 0
denoise_savepath = r'output\denoised'
# 图像修复
is_inpaint = False
inpaint_iters = 5
inpaint_runtime_list = ['onnxruntime','tensorrt','paddle_inference','paddle_serving']
inpaint_model_list = ['EUGAN']
inpaint_weightpath = r'models\inpaint\EUGAN.onnx'
inpaint_savepath = r'output\inpainted'

# 性状计算
is_calculate = True
# caculate_inputpath = inpaint_savepath
calculate_inputpath = r'output\inpainted'
calculate_savepath = r'output\calculated'




# trained_models = ['esegformer_b0', 'esegformer_b1', 'esegformer_b2', 'esegformer_b3', 'esegformer_b4', 'esegformer_b5',
#                   'deeplabv3_resnet50', 'deeplabv3_resnet101', 'deeplabv3p_resnet50', 'deeplabv3p_resnet101', 'hardnet',
#                   'pspnet_resnet50', 'pspnet_resnet101', 'segformer_b0', 'segformer_b1', 'segformer_b2', 'segformer_b3',
#                   'segformer_b4', 'segformer_b5', 'segnet', 'unet', 'unet_3plus', 'unet_plusplus']

# models = ['esegformer_b0', 'esegformer_b1', 'esegformer_b2', 'esegformer_b3', 'esegformer_b4', 'esegformer_b5',
#           'deeplabv3_resnet50', 'deeplabv3_resnet101', 'deeplabv3p_resnet50', 'deeplabv3p_resnet101', 'hardnet',
#           'pspnet_resnet50', 'pspnet_resnet101', 'segformer_b0', 'segformer_b1', 'segformer_b2', 'segformer_b3',
#           'segformer_b4', 'segformer_b5', 'segnet', 'unet', 'unet_3plus', 'unet_plusplus']
