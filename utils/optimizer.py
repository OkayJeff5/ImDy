import torch.optim as optim


def build_optimizer(net, config):
    param_group = [{'params': [p for p in net.parameters() if p.requires_grad], 'lr': config.OPTIMIZER.LR.base}]
    if config.OPTIMIZER.TYPE.upper() == 'SGD':
        optimizer = optim.SGD(param_group, lr=config.OPTIMIZER.LR.base,
                              momentum=config.OPTIMIZER.momentum,
                              weight_decay=config.OPTIMIZER.weight_decay)
    elif config.OPTIMIZER.TYPE.upper() == 'ADAM':
        optimizer = optim.Adam(param_group, lr=config.OPTIMIZER.LR.base,
                               weight_decay=config.OPTIMIZER.weight_decay,
                               eps=config.OPTIMIZER.EPS)
    elif config.OPTIMIZER.TYPE.upper() == 'ADAMW':
        optimizer = optim.AdamW(param_group, lr=config.OPTIMIZER.LR.base,
                                weight_decay=config.OPTIMIZER.weight_decay,
                                eps=config.OPTIMIZER.EPS)
    else:
        raise NotImplementedError
    return optimizer


def build_lr_scheduler(config, optimizer):

    if config.OPTIMIZER.LR.get("scheduler", False):
        sched_type = config.OPTIMIZER.LR.scheduler.get("type", "step")
        if sched_type == "cosine":
            lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config.OPTIMIZER.LR.scheduler.cosine.RESTART_PERIOD, T_mult=config.OPTIMIZER.LR.scheduler.cosine.RESTART_MULT, eta_min=config.OPTIMIZER.LR.scheduler.cosine.LEARNING_RATE_MIN)
        elif sched_type == "step":
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, config.OPTIMIZER.LR.scheduler.step.step_size, gamma=config.OPTIMIZER.LR.scheduler.step.gamma)
        else:
            raise NotImplementedError
    else:
        lr_scheduler = None

    return lr_scheduler
