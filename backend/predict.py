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
from PIL import Image
from torchvision import transforms as trans
from dataprocess.data.data_config import load_train_set, load_test_set, json_path, batch_size, image_size
import torch.nn.functional as F
from vis_feamap import show_feature_map
warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES

def predict(model, image_path):

    model.eval()

    # 开始训练
    # train(model=model,
    #       trainloader=trainloader,
    #       evalloader=evalloader,
    #       criterion=criterion,
    #       optimizer=optimizer,
    #       schedule=schedule,
    #       save_path=log_save_path,
    #       start_epoch=start_epoch,
    #       end_epoch=end_epoch)

    transform = trans.Compose([
                trans.Resize((image_size, image_size)),
                trans.ToTensor(),
                trans.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    test_image = Image.open(image_path).convert('RGB')
    test_image = transform(test_image).unsqueeze(0)
    test_image = test_image.to("cuda")
    with torch.no_grad():
        proposalN_windows_scores, proposalN_windows_logits, proposalN_indices, \
        window_scores, local_logits, srm_image, win_image, win_srm = model(test_image, "print")
    predict = F.softmax(local_logits, dim=-1)[:, 1]
    pred = local_logits.max(1, keepdim=True)[1].cpu()
    # print(predict.cpu())
    # print(pred)
    # print(proposalN_windows_score,proposalN_windows_logits, indices, \
    #         window_scores, local_logits, srm_image)
    return predict, pred
if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.cuda.set_device(0)  # 设置使用 GPU 0
