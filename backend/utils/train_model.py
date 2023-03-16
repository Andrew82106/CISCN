import os
import torch
from tqdm import tqdm
from tensorboardX import SummaryWriter
from config import proposalN, checkpoint_save_path
from utils.eval_model import eval
import warnings
import logging
from torch import nn
from utils.lr_schedule import adjust_lr
from utils.circle_loss import CircleLoss, convert_label_to_similarity
from utils.single_center_loss import SingleCenterLoss

warnings.filterwarnings("ignore")
def train(model,
          trainloader,
          evalloader,
          criterion,
          optimizer,
          schedule,
          save_path,
          start_epoch,
          end_epoch):

    best_AUC = 0.0
    best_epoch = 0
    # metric_criterion = SingleCenterLoss(m=0.3, D=2048)
    metric_criterion = CircleLoss(m=0.25, gamma=1)
    for epoch in range(start_epoch + 1, end_epoch + 1):
        model.train()
        print('Training %d epoch' % epoch)
        main_loss_sum = 0
        windowscls_loss_sum = 0
        windowscls_metric_loss_sum = 0
        main_metric_loss_sum = 0
        total_loss_sum = 0
        local_correct = 0
        logging.basicConfig(
            filename=os.path.join(save_path, 'train.log'),
            filemode='a',
            format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s',
            level=logging.INFO,
            force=True)
        warnings.filterwarnings("ignore")

        for i, data in enumerate(tqdm(trainloader)):
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            iterations = (epoch-1) * len(trainloader) + i
            adjust_lr(iterations, optimizer, schedule)
            optimizer.zero_grad()

            proposalN_windows_score, proposalN_windows_logits, indices, \
            window_scores, main_logits, _, window_embeddings, main_embeddings = model(images, 'train')
            main_loss = criterion(main_logits, labels)

            windowscls_loss = criterion(proposalN_windows_logits,
                               labels.unsqueeze(1).repeat(1, proposalN).view(-1))
            # windowscls_metric_loss = metric_criterion(nn.functional.normalize(window_embeddings),
            #                                           labels.unsqueeze(1).repeat(1, proposalN).view(-1))
            windowscls_metric_loss = metric_criterion(*convert_label_to_similarity(nn.functional.normalize(window_embeddings),
                                                       labels.unsqueeze(1).repeat(1, proposalN).view(-1)))

            if epoch < 2:
                # total_loss = main_loss + main_metric_loss
                total_loss = main_loss
            else:
                total_loss = main_loss + windowscls_loss + windowscls_metric_loss
                # total_loss = main_loss + main_metric_loss + windowscls_loss + windowscls_metric_loss
            main_loss_sum += main_loss.item()
            # main_metric_loss_sum += main_metric_loss.item()
            windowscls_loss_sum += windowscls_loss.item()
            windowscls_metric_loss_sum += windowscls_metric_loss.item()
            total_loss_sum += total_loss.item()
            pred = main_logits.max(1, keepdim=True)[1]
            local_correct += pred.eq(labels.view_as(pred)).sum().item()
            total_loss.backward()

            optimizer.step()

        main_loss_avg = main_loss_sum / (i + 1)
        main_metric_loss_avg = main_metric_loss_sum / (i + 1)
        windowscls_loss_avg = windowscls_loss_sum / (i + 1)
        windowscls_metric_loss_avg = windowscls_metric_loss_sum / (i + 1)
        total_loss_avg = total_loss_sum / (i + 1)
        local_accuracy = local_correct / len(trainloader.dataset)
        info = []
        phase = 'train'
        info.append('epoch:%s' % epoch)
        info.append('main_loss_avg:%.5f' % main_loss_avg)
        info.append('main_metric_loss_avg:%.5f' % main_metric_loss_avg)
        info.append('windowscls_loss_avg:%.5f' % windowscls_loss_avg)
        info.append('windowscls_metric_loss_avg:%.5f' % windowscls_metric_loss_avg)
        info.append('total_loss_avg:%.3f' % total_loss_avg)
        info.append('accuracy:%.3f%%' % (local_accuracy * 100.0))
        logging.info('{}: {}'.format(phase, '  '.join(info)))

        print(
            'Training: accuracy: {:.3f}%, loss: {:.5f}'.format(
                100. * local_accuracy, main_loss_avg))

        # eval testset
        r_acc, f_acc, local_accuracy, main_loss_avg, local_auc\
            = eval(model, evalloader, criterion, 'test', save_path, epoch)
        #
        print(
            'Testing: accuracy: {:.3f}%, auc: {:.5f}'.format(
                100. * local_accuracy, local_auc))

        # tensorboard
        with SummaryWriter(log_dir=os.path.join(save_path, 'log'), comment='eval') as writer:
            writer.add_scalar('Test/accuracy', local_accuracy, epoch)
            writer.add_scalar('Test/auc', local_auc, epoch)
            writer.add_scalar('Test/loss_avg', main_loss_avg, epoch)
            writer.add_scalar('Test/windowscls_loss_avg', r_acc, epoch)
            writer.add_scalar('Test/total_loss_avg', f_acc, epoch)

        # save checkpoint
        if os.path.exists(checkpoint_save_path) == False:
            os.makedirs(checkpoint_save_path)
        model_to_save = model.module if hasattr(model, "module") else model
        checkpoint = {"model": model_to_save.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}
        torch.save(checkpoint, checkpoint_save_path + "last.pt")
        if best_AUC < local_auc:
            best_AUC = local_auc
            best_epoch = epoch
            torch.save(checkpoint, checkpoint_save_path + "best.pt")
        print('current best epoch: %s ; current best AUC: %5f' % (best_epoch, best_AUC))
    os.remove(os.path.join(checkpoint_save_path, "last.pt"))