# -- coding: utf-8 --
import torch
import os
from PIL import Image
from torchvision.transforms.functional import normalize, resize, to_pil_image, to_tensor
from tensorboardX import SummaryWriter
import numpy as np
from utils.vis import image_with_boxes
from networks.model import MainNet
from config import num_classes, proposalN, channels, coordinates_cat


def windows(checkpoint_save_path, img_path, save_path):
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
    window_scores, local_logits, srm_image, win_image, win_srm = model(img_tensor, 'print')
    # print(win_image)
    # object branch tensorboard
    srm_image = srm_image * 25
    win_srm = win_srm * 25
    indices_ndarray = indices[:1, :proposalN].cpu().numpy()
    with SummaryWriter(comment='vis_windows object') as writer:
        box_raw_images = []
        raw_images = []
        box_srm_images = []
        srm_images = []
        win_images0 = []
        win_images1 = []
        win_images2 = []
        win_images3 = []
        win_images4 = []
        win_images5 = []
        win_srm0 = []
        win_srm1 = []
        win_srm2 = []
        win_srm3 = []
        win_srm4 = []
        win_srm5 = []
        for j, indice_ndarray in enumerate(indices_ndarray):
            box_raw_images.append(image_with_boxes(img_tensor[j], coordinates_cat[indice_ndarray]))
            raw_images.append(image_with_boxes(img_tensor[j]))
            box_srm_images.append(image_with_boxes(srm_image[j], coordinates_cat[indice_ndarray]))
            srm_images.append(image_with_boxes(srm_image[j]))
            win_images0.append(image_with_boxes(win_image[0]))
            win_images1.append(image_with_boxes(win_image[1]))
            win_images2.append(image_with_boxes(win_image[2]))
            win_images3.append(image_with_boxes(win_image[3]))
            win_images4.append(image_with_boxes(win_image[4]))
            win_images5.append(image_with_boxes(win_image[5]))
            win_srm0.append(image_with_boxes(win_srm[0]))
            win_srm1.append(image_with_boxes(win_srm[1]))
            win_srm2.append(image_with_boxes(win_srm[2]))
            win_srm3.append(image_with_boxes(win_srm[3]))
            win_srm4.append(image_with_boxes(win_srm[4]))
            win_srm5.append(image_with_boxes(win_srm[5]))
        box_raw_images = np.concatenate(box_raw_images, axis=1)
        raw_images = np.concatenate(raw_images, axis=1)
        box_srm_images = np.concatenate(box_srm_images, axis=1)
        srm_images = np.concatenate(srm_images, axis=1)
        win_images0 = np.concatenate(win_images0, axis=1)
        win_images1 = np.concatenate(win_images1, axis=1)
        win_images2 = np.concatenate(win_images2, axis=1)
        win_images3 = np.concatenate(win_images3, axis=1)
        win_images4 = np.concatenate(win_images4, axis=1)
        win_images5 = np.concatenate(win_images5, axis=1)
        win_srm0 = np.concatenate(win_srm0, axis=1)
        win_srm1 = np.concatenate(win_srm1, axis=1)
        win_srm2 = np.concatenate(win_srm2, axis=1)
        win_srm3 = np.concatenate(win_srm3, axis=1)
        win_srm4 = np.concatenate(win_srm4, axis=1)
        win_srm5 = np.concatenate(win_srm5, axis=1)
        import matplotlib.pyplot as plt

        plt.axis('off')
        fig = plt.gcf()
        fig.set_size_inches(7.0 / 3, 7.0 / 3)  # dpi = 300, output = 700*700 pixels
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        save_path_list = []
        plt.imshow(box_raw_images)
        plt.savefig(save_path + "/box_raw_images.png")
        save_path_list.append(save_path + "/box_raw_images.png")

        plt.imshow(raw_images)
        plt.savefig(save_path + "/raw_images.png")
        save_path_list.append(save_path + "/raw_images.png")

        plt.imshow(box_srm_images)
        plt.savefig(save_path + "/box_srm_images.png")
        save_path_list.append(save_path + "/box_srm_images.png")

        plt.imshow(srm_images)
        plt.savefig(save_path + "/srm_images.png")
        save_path_list.append(save_path + "/srm_images.png")

        plt.imshow(win_images0)
        plt.savefig(save_path + "/win_images0.png")
        save_path_list.append(save_path + "/win_images0.png")

        plt.imshow(win_images1)
        plt.savefig(save_path + "/win_images1.png")
        save_path_list.append(save_path + "/win_images1.png")

        plt.imshow(win_images2)
        plt.savefig(save_path + "/win_images2.png")
        save_path_list.append(save_path + "/win_images2.png")

        plt.imshow(win_images3)
        plt.savefig(save_path + "/win_images3.png")
        save_path_list.append(save_path + "/win_images3.png")

        plt.imshow(win_images4)
        plt.savefig(save_path + "/win_images4.png")
        save_path_list.append(save_path + "/win_images4.png")

        plt.imshow(win_images5)
        plt.savefig(save_path + "/win_images5.png")
        save_path_list.append(save_path + "/win_images5.png")

        plt.imshow(win_srm0)
        plt.savefig(save_path + "/win_srm0.png")
        save_path_list.append(save_path + "/win_srm0.png")

        plt.imshow(win_srm1)
        plt.savefig(save_path + "/win_srm1.png")
        save_path_list.append(save_path + "/win_srm1.png")

        plt.imshow(win_srm2)
        plt.savefig(save_path + "/win_srm2.png")
        save_path_list.append(save_path + "/win_srm2.png")

        plt.imshow(win_srm3)
        plt.savefig(save_path + "/win_srm3.png")
        save_path_list.append(save_path + "/win_srm3.png")

        plt.imshow(win_srm4)
        plt.savefig(save_path + "/win_srm4.png")
        save_path_list.append(save_path + "/win_srm4.png")

        plt.imshow(win_srm5)
        plt.savefig(save_path + "/win_srm5.png")
        save_path_list.append(save_path + "/win_srm5.png")

        return save_path_list

        # writer.add_images(forgery_type + '/' + image_id + '/' + 'raw_images_with_windows', box_raw_images, dataformats='HWC')
        # writer.add_images(forgery_type + '/' + image_id + '/' + 'raw_images', raw_images, dataformats='HWC')
        # writer.add_images(forgery_type + '/' + image_id + '/' + 'srm_images_with_windows', box_srm_images, dataformats='HWC')
        # writer.add_images(forgery_type + '/' + image_id + '/' + 'srm_images', srm_images, dataformats='HWC')
        # writer.add_images(forgery_type + '/' + image_id + '/' + 'win_images0', win_images0, dataformats='HWC')
        # writer.add_images(forgery_type + '/' + image_id + '/' + 'win_images1', win_images1, dataformats='HWC')
        # writer.add_images(forgery_type + '/' + image_id + '/' + 'win_images2', win_images2, dataformats='HWC')
        # writer.add_images(forgery_type + '/' + image_id + '/' + 'win_images3', win_images3, dataformats='HWC')
        # writer.add_images(forgery_type + '/' + image_id + '/' + 'win_images4', win_images4, dataformats='HWC')
        # writer.add_images(forgery_type + '/' + image_id + '/' + 'win_images5', win_images5, dataformats='HWC')
        # writer.add_images(forgery_type + '/' + image_id + '/' + 'win_srm0', win_srm0, dataformats='HWC')
        # writer.add_images(forgery_type + '/' + image_id + '/' + 'win_srm1', win_srm1, dataformats='HWC')
        # writer.add_images(forgery_type + '/' + image_id + '/' + 'win_srm2', win_srm2, dataformats='HWC')
        # writer.add_images(forgery_type + '/' + image_id + '/' + 'win_srm3', win_srm3, dataformats='HWC')
        # writer.add_images(forgery_type + '/' + image_id + '/' + 'win_srm4', win_srm4, dataformats='HWC')
        # writer.add_images(forgery_type + '/' + image_id + '/' + 'win_srm5', win_srm5, dataformats='HWC')

