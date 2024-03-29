import math
import numpy as np
from config import end_epoch, warmup_batchs, init_lr

def cosine_decay(batchs: int, decay_type: int = 1):
    total_batchs = end_epoch * batchs
    iters = np.arange(total_batchs - warmup_batchs)

    if decay_type == 1:
        schedule = np.array([1e-12 + 0.5 * (init_lr - 1e-12) * (1 + \
                             math.cos(math.pi * t / total_batchs)) for t in iters])
    elif decay_type == 2:
        schedule = init_lr * np.array([math.cos(7*math.pi*t / (16*total_batchs)) for t in iters])
    else:
        raise ValueError("Not support this deccay type")
    
    if warmup_batchs > 0:
        warmup_lr_schedule = np.linspace(1e-9, init_lr, warmup_batchs)
        schedule = np.concatenate((warmup_lr_schedule, schedule))

    return schedule

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        if param_group["lr"] is not None:
            return param_group["lr"]

def adjust_lr(iteration, optimizer, schedule):
    for param_group in optimizer.param_groups:
        param_group["lr"] = schedule[iteration]
