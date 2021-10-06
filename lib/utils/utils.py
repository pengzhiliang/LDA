import logging
import time
import os

import torch
from utils.lr_scheduler import WarmupMultiStepLR, WarmupCosineAnnealingLR
from net import Network


import pdb

def create_logger(cfg):
    dataset = cfg.DATASET.DATASET
    net_type = cfg.BACKBONE.TYPE
    module_type = cfg.MODULE.TYPE
    log_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    time_str = time.strftime("%Y-%m-%d-%H-%M")
    log_name = "{}_{}_{}_{}.log".format(dataset, net_type, module_type, time_str)
    log_file = os.path.join(log_dir, log_name)
    # set up logger
    print("=> creating log {}".format(log_file))
    head = "%(asctime)-15s %(message)s"
    logging.basicConfig(filename=str(log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger("").addHandler(console)

    logger.info("---------------------Cfg is set as follow--------------------")
    logger.info(cfg)
    logger.info("-------------------------------------------------------------")
    return logger, log_file

def create_valid_logger(cfg):
    dataset = cfg.DATASET.DATASET
    net_type = cfg.BACKBONE.TYPE
    module_type = cfg.MODULE.TYPE

    test_model_path = os.path.join(*cfg.TEST.MODEL_FILE.split('/')[:-2])
    log_dir = os.path.join(test_model_path, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    time_str = time.strftime("%Y-%m-%d-%H-%M")
    log_name = "Test_{}_{}_{}_{}.log".format(dataset, net_type, module_type, time_str)
    log_file = os.path.join(log_dir, log_name)
    # set up logger
    print("=> creating log {}".format(log_file))
    head = "%(asctime)-15s %(message)s"
    logging.basicConfig(filename=str(log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger("").addHandler(console)

    logger.info("---------------------Start Testing--------------------")
    logger.info("Test model: {}".format(cfg.TEST.MODEL_FILE))

    return logger, log_file


def get_optimizer(cfg, model, combiner=None, **kwargs):
    base_lr = cfg.TRAIN.OPTIMIZER.BASE_LR
    params = []

    for name, p in model.named_parameters():
        if p.requires_grad:
            params.append({"params": p})

    if cfg.TRAIN.OPTIMIZER.TYPE == "SGD":
        optimizer = torch.optim.SGD(
            params,
            lr=base_lr,
            momentum=cfg.TRAIN.OPTIMIZER.MOMENTUM,
            weight_decay=cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY,
            nesterov=True,
        )
    elif cfg.TRAIN.OPTIMIZER.TYPE == 'SGDWithExtraWeightDecay':
        optimizer = SGDWithExtraWeightDecay(
            params,
            kwargs['num_class_list'],
            kwargs['classifier_shape'],
            lr=base_lr,
            momentum=cfg.TRAIN.OPTIMIZER.MOMENTUM,
            weight_decay=cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY,
            nesterov=True,
        )
    elif cfg.TRAIN.OPTIMIZER.TYPE == "ADAM":
        optimizer = torch.optim.Adam(
            params,
            lr=base_lr,
            betas=(0.9, 0.999),
            weight_decay=cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY,
        )
    return optimizer


def get_scheduler(cfg, optimizer):
    if cfg.TRAIN.LR_SCHEDULER.TYPE == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            cfg.TRAIN.LR_SCHEDULER.LR_STEP,
            gamma=cfg.TRAIN.LR_SCHEDULER.LR_FACTOR,
        )
    elif cfg.TRAIN.LR_SCHEDULER.TYPE == "cosine":
        if cfg.TRAIN.LR_SCHEDULER.COSINE_DECAY_END > 0:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.TRAIN.LR_SCHEDULER.COSINE_DECAY_END, eta_min=cfg.TRAIN.LR_SCHEDULER.COSINE_DECAY_END
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.TRAIN.MAX_EPOCH, eta_min=0.0
            )
    elif cfg.TRAIN.LR_SCHEDULER.TYPE == "warmup":
        scheduler = WarmupMultiStepLR(
            optimizer,
            cfg.TRAIN.LR_SCHEDULER.LR_STEP,
            gamma=cfg.TRAIN.LR_SCHEDULER.LR_FACTOR,
            warmup_epochs=cfg.TRAIN.LR_SCHEDULER.WARM_EPOCH,
        )
    elif cfg.TRAIN.LR_SCHEDULER.TYPE == 'warmupcosine':
        scheduler = WarmupCosineAnnealingLR(
            optimizer,
            T_max=cfg.TRAIN.MAX_EPOCH,
            eta_min=1e-4,
            warmup_epochs=cfg.TRAIN.LR_SCHEDULER.WARM_EPOCH,
        )
    else:
        raise NotImplementedError("Unsupported LR Scheduler: {}".format(cfg.TRAIN.LR_SCHEDULER.TYPE))

    return scheduler


def get_model(cfg, num_classes, device, logger):
    model = Network(cfg, logger=logger, mode="train", num_classes=num_classes)

    if cfg.BACKBONE.FREEZE == True:
        model.freeze_backbone()
        logger.info("Backbone has been freezed")
        if cfg.BACKBONE.REFEEZE_LAST_BLOCK:
            model.refreeze_last_block()
            logger.info("The last block of backbone has been refreezed")
        if cfg.BACKBONE.REFEEZE_LAST_STAGE:
            model.refreeze_last_stage()
            logger.info("The last stage of backbone has been refreezed")

    if cfg.CPU_MODE:
        model = model.to(device)
    else:
        model = torch.nn.DataParallel(model).cuda()

    return model

def get_category_list(annotations, num_classes, cfg):
    num_list = [0] * num_classes
    cat_list = []
    print("Weight List has been produced")
    for anno in annotations:
        category_id = anno["category_id"]
        num_list[category_id] += 1
        cat_list.append(category_id)
    return num_list, cat_list

class _RequiredParameter(object):
    """Singleton class representing a required parameter for an Optimizer."""
    def __repr__(self):
        return "<required parameter>"

required = _RequiredParameter()

class SGDWithExtraWeightDecay(torch.optim.Optimizer):

    def __init__(self, params, num_class_list, classifier_shape, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):

        self.extra_weight_decay = weight_decay / num_class_list[:, None].repeat(1, classifier_shape[-1])
        self.classifier_shape = classifier_shape
        self.first = True

        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGDWithExtraWeightDecay, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGDWithExtraWeightDecay, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                    if self.classifier_shape == d_p.shape:
                        if self.first:
                            self.first = False
                        else:
                            d_p.add_(self.extra_weight_decay * p.data)
                            self.first = True

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss