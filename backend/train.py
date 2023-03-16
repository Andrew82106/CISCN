#coding=utf-8
import torch
import shutil
import time
from config import num_classes, \
    end_epoch, init_lr, CUDA_VISIBLE_DEVICES, \
    proposalN, channels, log_save_path, checkpoint_save_path
from utils.train_model import train
from utils.lr_schedule import cosine_decay
from networks.model import MainNet
from dataprocess.data.dataloader import read_dataloader
import os
import warnings
from torch import nn
import numpy as np
warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES

def main():

    #加载数据
    trainloader = read_dataloader("train")
    evalloader = read_dataloader("eval")

    #定义模型
    model = MainNet(proposalN=proposalN, num_classes=num_classes, channels=channels)

    #设置训练参数
    criterion = nn.CrossEntropyLoss()

    #加载checkpoint
    if os.path.exists(checkpoint_save_path + "last.pt"):
        checkpoint = torch.load(checkpoint_save_path + "last.pt", map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0

    # define optimizers
    optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr)
    schedule = cosine_decay(len(trainloader))
    model = model.cuda()  # 部署在GPU

    # 保存config参数信息
    # time_str = time.strftime("%Y%m%d-%H%M%S")
    if os.path.exists(log_save_path) == False:
        os.makedirs(log_save_path)
    # shutil.copy('./config.py', os.path.join(log_save_path, "{}config.py".format(time_str)))

    # 开始训练
    train(model=model,
          trainloader=trainloader,
          evalloader=evalloader,
          criterion=criterion,
          optimizer=optimizer,
          schedule=schedule,
          save_path=log_save_path,
          start_epoch=start_epoch,
          end_epoch=end_epoch)


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.cuda.set_device(0)  # 设置使用 GPU 0
    main()
