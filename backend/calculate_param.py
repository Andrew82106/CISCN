from torchstat import stat
from networks.model import MainNet
from config import num_classes, \
    end_epoch, init_lr, CUDA_VISIBLE_DEVICES, \
    proposalN, channels, log_save_path, checkpoint_save_path

model = MainNet(proposalN=proposalN, num_classes=num_classes, channels=channels)
stat(model, (3, 224, 224))
