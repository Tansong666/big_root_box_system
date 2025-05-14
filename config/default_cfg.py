# 当前工作目录的output目录设为根目录
root_path = r'E:\big_root_system'

# concat
data_path =r'data'
concat_x1 = 2000
concat_x2 = 2700
concat_x3 = 3350
concat_savepath = r'output\concated'

# 必选参数edge和serving二选一
# is_Edge=False
is_Serving=False

# seg
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


# post_savepath = r'output\post'

# calculate_savepath = r'output\calculate'




# trained_models = ['esegformer_b0', 'esegformer_b1', 'esegformer_b2', 'esegformer_b3', 'esegformer_b4', 'esegformer_b5',
#                   'deeplabv3_resnet50', 'deeplabv3_resnet101', 'deeplabv3p_resnet50', 'deeplabv3p_resnet101', 'hardnet',
#                   'pspnet_resnet50', 'pspnet_resnet101', 'segformer_b0', 'segformer_b1', 'segformer_b2', 'segformer_b3',
#                   'segformer_b4', 'segformer_b5', 'segnet', 'unet', 'unet_3plus', 'unet_plusplus']

# models = ['esegformer_b0', 'esegformer_b1', 'esegformer_b2', 'esegformer_b3', 'esegformer_b4', 'esegformer_b5',
#           'deeplabv3_resnet50', 'deeplabv3_resnet101', 'deeplabv3p_resnet50', 'deeplabv3p_resnet101', 'hardnet',
#           'pspnet_resnet50', 'pspnet_resnet101', 'segformer_b0', 'segformer_b1', 'segformer_b2', 'segformer_b3',
#           'segformer_b4', 'segformer_b5', 'segnet', 'unet', 'unet_3plus', 'unet_plusplus']
