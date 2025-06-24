# 必选参数edge和serving二选一

is_process = False

# is_Edge=False
is_Serving=False

# 当前工作目录的output目录设为根目录
root_path = r'E:\big_root_system'
data_path =r'data'
input_path = None
# concat
is_concat = True
concat_x1 = 2000
concat_x2 = 2700
concat_x3 = 3350
# concat_savepath = r'output\concated'
concat_savepath = r'output\test\concated'

# seg
is_seg = True
seg_runtime_list = ['paddle_inference','onnxruntime','tensorrt','paddle_serving']
seg_model_list = ['segformer','deeplabv3','hardnet','unet']
is_slide=True
crop_size = 768
stride = 768
is_resize = True
resize_scale = 1.0
# seg_path
# seg_weightpath = r'models\segment\bigbox_segformer'
seg_weightpath = r'models\segment\pp_liteseg_stdc1-paddle'
# seg_imagepath = r'output\concated'
# seg_savepath = r'output\segmented'
seg_savepath = r'output\test\segmented\ppliteseg1'


#### 后处理
# 图像去噪
is_denoise = True
# post_inputpath = r'output\test\0510'
is_rsa = False
dilate_iters = 10
threshold_area = 1000
is_rba = False
rba_left = 0
rba_right = 0
rba_top = 0
rba_bottom = 0
# denoise_savepath = r'output\denoised'
denoise_savepath = r'output\test\denoised'

# 图像修复
is_inpaint = True
inpaint_iters = 5
inpaint_runtime_list = ['onnxruntime','tensorrt','paddle_inference','paddle_serving']
inpaint_model_list = ['EUGAN']
inpaint_weightpath = r'models\inpaint\EUGAN.onnx'
# inpaint_savepath = r'output\inpainted'
inpaint_savepath = r'output\test\inpainted'

# 性状计算
is_calculate = False
# caculate_inputpath = inpaint_savepath
# calculate_inputpath = r'output\inpainted'
# calculate_savepath = r'output\calculated'
calculate_savepath = r'output\test\calculated'





# trained_models = ['esegformer_b0', 'esegformer_b1', 'esegformer_b2', 'esegformer_b3', 'esegformer_b4', 'esegformer_b5',
#                   'deeplabv3_resnet50', 'deeplabv3_resnet101', 'deeplabv3p_resnet50', 'deeplabv3p_resnet101', 'hardnet',
#                   'pspnet_resnet50', 'pspnet_resnet101', 'segformer_b0', 'segformer_b1', 'segformer_b2', 'segformer_b3',
#                   'segformer_b4', 'segformer_b5', 'segnet', 'unet', 'unet_3plus', 'unet_plusplus']

# models = ['esegformer_b0', 'esegformer_b1', 'esegformer_b2', 'esegformer_b3', 'esegformer_b4', 'esegformer_b5',
#           'deeplabv3_resnet50', 'deeplabv3_resnet101', 'deeplabv3p_resnet50', 'deeplabv3p_resnet101', 'hardnet',
#           'pspnet_resnet50', 'pspnet_resnet101', 'segformer_b0', 'segformer_b1', 'segformer_b2', 'segformer_b3',
#           'segformer_b4', 'segformer_b5', 'segnet', 'unet', 'unet_3plus', 'unet_plusplus']
