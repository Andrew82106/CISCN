from utils.indices2coordinates import indices2coordinates
from utils.compute_window_nums import compute_window_nums
import numpy as np

CUDA_VISIBLE_DEVICES = '0'  # The current version only supports one GPU training

model_name = 'tsnet-829'

training_set = "FF++"
# FF++ celeb-df-v2
training_forgery_type = "all"
# deepfakes neuraltextures face2face faceshifter faceswap all testdata None
training_compression = "c23"
# c23 c40 None

testing_set = "celeb-df-v2"
# FF++ celeb-df-v2
testing_forgery_type = "None"
# deepfakes neuraltextures face2face faceshifter faceswap all testdata None
testing_compression = "None"
# c23 c40 None

num_classes = 2
batch_size = 12
num_workers = 4
vis_num = batch_size  # The number of visualized images in tensorboard
end_epoch = 20
init_lr = 0.0002
# init_lr = 1e-4
warmup_batchs = 800

log_save_path = './output/log/{0}_{1}({2})--{3}_{4}({5})/{6}/' .format(training_set, training_forgery_type, training_compression, testing_set, testing_forgery_type, testing_compression, model_name)
checkpoint_save_path = './output/checkpoint/{0}_{1}({2})--{3}_{4}({5})/{6}/' .format(training_set, training_forgery_type, training_compression, testing_set, testing_forgery_type, testing_compression, model_name)

weight_decay = 1e-4
stride = 32
channels = 2048
input_size = 320

# windows info for CAR and Aircraft
N_list = [6]
proposalN = sum(N_list)  # proposal window num
iou_threshs = [0.25, 0.25, 0.25]

# ratios = [[6, 6], [5, 7], [7, 5],
#           [8, 8], [6, 10], [10, 6], [7, 9], [9, 7],
#           [10, 10], [9, 11], [11, 9], [8, 12], [12, 8]]

ratios = [[4, 4], [5, 5], [4, 6], [6, 4], [6, 6], [5, 7], [7, 5],
          [8, 8], [6, 10], [10, 6], [7, 9], [9, 7]]

'''indice2coordinates'''
window_nums = compute_window_nums(ratios, stride, input_size)
indices_ndarrays = [np.arange(0,window_num).reshape(-1,1) for window_num in window_nums]
coordinates = [indices2coordinates(indices_ndarray, stride, input_size, ratios[i]) for i, indices_ndarray in enumerate(indices_ndarrays)] # 每个window在image上的坐标
coordinates_cat = np.concatenate(coordinates, 0)
window_milestones = [sum(window_nums[:i+1]) for i in range(len(window_nums))]
# window_nums_sum = [0, sum(window_nums[:3]), sum(window_nums[3:8]), sum(window_nums[8:])]
window_nums_sum = [0, sum(window_nums[:])]

aug_test = False
