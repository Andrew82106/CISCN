import torch
from tqdm import tqdm
import os
from tensorboardX import SummaryWriter
import numpy as np
from config import coordinates_cat, proposalN, vis_num
from utils.vis import image_with_boxes
import logging
import warnings
import torch.nn.functional as F
from sklearn.metrics import roc_curve, accuracy_score
from sklearn.metrics import auc as cal_auc

def eval(model, evalloader, criterion, status, save_path, epoch):
    model.eval()
    print('Evaluating')

    ce_loss_sum = 0
    correct = 0

    logging.basicConfig(
        filename=os.path.join(save_path, 'eval.log'),
        filemode='a',
        format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s',
        level=logging.INFO,
        force=True)
    warnings.filterwarnings("ignore")
    with torch.no_grad():
        y_true, y_pred = [], []
        for i, data in enumerate(tqdm(evalloader)):
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()

            proposalN_windows_score,proposalN_windows_logits, indices, \
            window_scores, local_logits, srm_image = model(images, status)

            ce_loss = criterion(local_logits, labels)

            ce_loss_sum += ce_loss.item()

            labels = labels.cpu()
            y_true.extend(labels)
            predict = F.softmax(local_logits, dim=-1)[:, 1]
            y_pred.extend(predict.cpu())


            pred = local_logits.max(1, keepdim=True)[1].cpu()
            correct += pred.eq(labels.view_as(pred)).sum().item()

            # object branch tensorboard
            if i == 0:
                indices_ndarray = indices[:vis_num, :proposalN].cpu().numpy()
                with SummaryWriter(log_dir=os.path.join(save_path, 'log'), comment=status + 'object') as writer:
                    box_raw_images = []
                    raw_images = []
                    box_srm_images = []
                    srm_images = []
                    for j, indice_ndarray in enumerate(indices_ndarray):
                        box_raw_images.append(image_with_boxes(images[j], coordinates_cat[indice_ndarray]))
                        raw_images.append(image_with_boxes(images[j]))
                        box_srm_images.append(image_with_boxes(srm_image[j], coordinates_cat[indice_ndarray]))
                        srm_images.append(image_with_boxes(srm_image[j]))
                    box_raw_images = np.concatenate(box_raw_images, axis=1)
                    raw_images = np.concatenate(raw_images, axis=1)
                    box_srm_images = np.concatenate(box_srm_images, axis=1)
                    srm_images = np.concatenate(srm_images, axis=1)
                    writer.add_images(status + '/' + 'raw_images_with_windows', box_raw_images, epoch, dataformats='HWC')
                    writer.add_images(status + '/' + 'raw_images', raw_images, epoch, dataformats='HWC')
                    writer.add_images(status + '/' + 'srm_images_with_windows', box_srm_images, epoch, dataformats='HWC')
                    writer.add_images(status + '/' + 'srm_images', srm_images, epoch, dataformats='HWC')

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    AUC = cal_auc(fpr, tpr)

    idx_real = np.where(y_true == 1)[0]
    idx_fake = np.where(y_true == 0)[0]

    r_acc = accuracy_score(y_true[idx_real], y_pred[idx_real] > 0.5)
    f_acc = accuracy_score(y_true[idx_fake], y_pred[idx_fake] > 0.5)

            # if status == 'train':
            #     if i >= 2 :
            #         break
    loss_avg = ce_loss_sum / (i + 1)
    accuracy = correct / len(evalloader.dataset)

    info = []
    phase = 'eval'
    info.append('epoch:%s' % epoch)
    info.append('loss_avg:%.5f' % loss_avg)
    info.append('accuracy:%.3f%%' % (accuracy * 100.0))
    info.append('auc:%.5f' % AUC)
    info.append('r_acc:%.3f%%' % (r_acc * 100.0))
    info.append('f_acc:%.3f%%' % (f_acc * 100.0))
    logging.info('{}: {}'.format(phase, '  '.join(info)))

    return r_acc, f_acc, accuracy, loss_avg, AUC