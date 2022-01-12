from torch import optim
from enum import Enum


class OptimizerType(Enum):
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"

def get_optimizer(optimizer_type, model_params, learning_rate):

    params = filter(lambda p: p.requires_grad, model_params)
    optimizer_args = (params, learning_rate)

    if optimizer_type == OptimizerType.ADAM.value:
        return optim.Adam(*optimizer_args)
    elif optimizer_type == OptimizerType.ADAMW.value:
        return optim.AdamW(*optimizer_args)
    elif optimizer_type == OptimizerType.SGD.value:
        return optim.SGD(*optimizer_args)
    else:
        raise ValueError("invalid optimizer name")

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)
