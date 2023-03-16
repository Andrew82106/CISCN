import torch
import os
from PIL import Image
from torchvision.transforms.functional import normalize, resize, to_pil_image, to_tensor
from networks.model import MainNet
from config import num_classes, proposalN, channels, coordinates_cat
import imageio
import matplotlib.pyplot as plt
import numpy as np

#  可视化特征图
def show_feature_map(feature_map):  # feature_map=torch.Size([1, 64, 55, 55]),feature_map[0].shape=torch.Size([64, 55, 55])
    # feature_map[2].shape     out of bounds
    feature_map = feature_map.squeeze(0)  # 压缩成torch.Size([64, 55, 55])

    # 以下4行，通过双线性插值的方式改变保存图像的大小
    feature_map = feature_map.view(1, feature_map.shape[0], feature_map.shape[1], feature_map.shape[2])  # (1,64,55,55)
    upsample = torch.nn.UpsamplingBilinear2d(size=(320, 320))  # 这里进行调整大小
    feature_map = upsample(feature_map)
    feature_map = feature_map.view(feature_map.shape[1], feature_map.shape[2], feature_map.shape[3])

    feature_map_num = feature_map.shape[0]  # 返回通道数
    row_num = int(np.ceil(np.sqrt(feature_map_num)))  # 8
    plt.figure()
    for index in range(1, feature_map_num + 1):  # 通过遍历的方式，将64个通道的tensor拿出
        plt.subplot(row_num, row_num, index)
        plt.imshow(feature_map[index - 1].cpu().detach().numpy(), cmap='gray')  # feature_map[0].shape=torch.Size([55, 55])
        # 将上行代码替换成，可显示彩色 plt.imshow(transforms.ToPILImage()(feature_map[index - 1]))#feature_map[0].shape=torch.Size([55, 55])
        plt.axis('off')
        # imageio.imsave(r'C:\Users\Administrator\Desktop\temp\srmatt12\No.' + str(index) + ".png", (feature_map[index - 1].sigmoid().cpu().detach().numpy()*255).astype(np.uint8))
    plt.show()

def vis_featuremap(img_path, forgery_type, image_id, log_save_path, checkpoint_save_path):
    print('Visualizing')
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = MainNet(proposalN=proposalN, num_classes=num_classes, channels=channels)

    if os.path.exists(checkpoint_save_path + "best.pt"):
        checkpoint = torch.load(checkpoint_save_path + "best.pt", map_location=torch.device('cpu'))
        print("successfully loading", checkpoint_save_path, "best.pt")
        model.load_state_dict(checkpoint['model'])
    model = model.to(device=device)  # 部署在GPU
    model.eval()
    pil_img = Image.open(img_path, mode="r").convert("RGB")
    img_tensor = normalize(to_tensor(resize(pil_img, (320, 320))), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).to(
        device=device
    )
    img_tensor = img_tensor.unsqueeze(0)
    proposalN_windows_score, proposalN_windows_logits, indices, \
    window_scores, local_logits, srm_image, srmatt11, srmatt12, srmatt21, srmatt22 = model(img_tensor, 'print')
    show_feature_map(srmatt12)



if __name__ == "__main__":
    img_path = r"G:\CISCN作品赛\processeddataset\FF++\fake\deepfakes\c23\1.png"
    #1 266 512 618 942
    model_name = 'tsnet-726'
    forgery_type = "youtube"
    image_id = "942"

    training_set = "FF++"
    # FF++ celeb-df-v2
    training_forgery_type = "all"
    # deepfakes neuraltextures face2face faceshifter faceswap all testdata None
    training_compression = "c23"
    # c23 c40 None
    testing_set = "FF++"
    # FF++ celeb-df-v2
    testing_forgery_type = "all"
    # deepfakes neuraltextures face2face faceshifter faceswap all testdata None
    testing_compression = "c23"

    # c23 c40 None
    log_save_path = r"C:\Users\Administrator\Desktop\vis_windows"
    checkpoint_save_path = './output/checkpoint/{0}_{1}({2})--{3}_{4}({5})/{6}/'.format(training_set,
                                                                                        training_forgery_type,
                                                                                        training_compression,
                                                                                        testing_set,
                                                                                        testing_forgery_type,
                                                                                        testing_compression, model_name)

    vis_featuremap(img_path, forgery_type, image_id, log_save_path, checkpoint_save_path)