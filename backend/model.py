# -- coding: utf-8 --
import torch
from config import num_classes, \
    init_lr, CUDA_VISIBLE_DEVICES, \
    proposalN, channels
from networks.model import MainNet
import os
import warnings
from torch import nn
import numpy as np
warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES

def model(checkpoint_save_path):
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #定义模型
    model = MainNet(proposalN=proposalN, num_classes=num_classes, channels=channels)

    #设置训练参数
    criterion = nn.CrossEntropyLoss()

    #加载checkpoint
    checkpoint = torch.load(checkpoint_save_path + "best.pt", map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model"])


    # define optimizers
    optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr)
    # schedule = cosine_decay(len(trainloader))
    model = model.cuda()  # 部署在GPU

    return model
if __name__ == '__main__':
    # torch.cuda.set_device(0)  # 设置使用 GPU 0
    model_name = 'tsnet-815'

    training_set = "FF++"
    # FF++ celeb-df-v2
    training_forgery_type = "deepfakes"
    # deepfakes neuraltextures face2face faceshifter faceswap all testdata None
    training_compression = "c23"
    # c23 c40 None
    testing_set = "FF++"
    # FF++ celeb-df-v2
    testing_forgery_type = "deepfakes"
    # deepfakes neuraltextures face2face faceshifter faceswap all testdata None
    testing_compression = "c23"
    # c23 c40 None
    checkpoint_save_path = 'G:/CISCN作品赛/CISCN/backend/output/checkpoint/{0}_{1}({2})--{3}_{4}({5})/{6}/'.format(
        training_set,
        training_forgery_type,
        training_compression,
        testing_set,
        testing_forgery_type,
        testing_compression,
        model_name)
    model = model(checkpoint_save_path)