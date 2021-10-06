import _init_paths
from core.evaluate import accuracy, AverageMeter, FusionMatrix, AverageMeterList, accuracy_shot, accuracy_shot_perclass

import numpy as np
import torch
import time

import pdb


def backhook(grad):
    print(grad)


def train_model(
        trainLoader,
        model,
        epoch,
        epoch_number,
        optimizer,
        combiner,
        criterion,
        cfg,
        logger,
        **kwargs
):
    if cfg.EVAL_MODE:
        model.eval()
    else:
        model.train()


    start_time = time.time()
    number_batch = len(trainLoader)

    if cfg.CLASSIFIER.TYPE == 'LDA':
        all_loss = [AverageMeter(), AverageMeter()]
        acc = [AverageMeterList(4), AverageMeterList(4)]
    else:
        all_loss = AverageMeter()
        acc = AverageMeter()

    cur_lr = optimizer.param_groups[0]['lr']
    logger.info("\n==========> Epoch: {:>3d}/{} LR:{:>1.5f}".format(epoch, epoch_number, cur_lr))
    for i, (image, label, meta) in enumerate(trainLoader):

        cnt = label.shape[0]
        res = combiner.forward(model, criterion, image, label, meta, logger=logger)

        if len(res) == 2:
            loss, now_acc = res
        else:
            loss, now_acc, now_cnt = res

        optimizer.zero_grad()
        if cfg.CLASSIFIER.TYPE == 'LDA':
            sum(loss).backward()
            all_loss[0].update(loss[0].data.item(), cnt)
            all_loss[1].update(loss[1].data.item(), cnt)
            acc[0].update(now_acc[0], now_cnt[0])
            acc[1].update(now_acc[1], now_cnt[1])
        else:
            loss.backward()  # retain_graph=True
            all_loss.update(loss.data.item(), cnt)
            acc.update(now_acc, cnt)

        optimizer.step()

        if i % cfg.SHOW_STEP == 0:
            if cfg.CLASSIFIER.TYPE == 'LDA':
                alpha = combiner.adap_alpha.get_act_alpha().item()
                delta_acc = combiner.adap_alpha.get_acc_diff.item()
                pbar_str = "Batch:{:>3d}/{} Adaptitve_Alpha:{:>3.3f} D_Acc:{:>2.2f} Loss.0:{:>5.3f} Loss.1:{:>5.3f} " \
                           "Acc.0:[r:{:>2.2f} c:{:>2.2f} f:{:>2.2f} a:{:>2.2f}] Acc.1:[r:{:>2.2f} c:{:>2.2f} f:{:>2.2f} a:{:>2.2f}]".format(
                    i, number_batch, alpha, delta_acc,
                    all_loss[0].val, all_loss[1].val,
                    acc[0].val[0] * 100, acc[0].val[1] * 100, acc[0].val[2] * 100, acc[0].val[3] * 100,
                    acc[1].val[0] * 100, acc[1].val[1] * 100,
                    acc[1].val[2] * 100, acc[1].val[3] * 100,
                )  # "aver_acc:[{:>2.2f} {:>2.2f}]" now_acc[1] * 100, now_acc[3] * 100,
            else:
                pbar_str = "Batch:{:>3d}/{} Batch_Loss:{:>5.3f}  Batch_Accuracy:{:>5.2f}%     ".format(
                    epoch, i, number_batch, optimizer.param_groups[0]['lr'], all_loss.val, acc.val * 100
                )
            logger.info(pbar_str)
    end_time = time.time()


    if cfg.CLASSIFIER.TYPE == 'LDA':
        epoch_delta_acc = (acc[0].avg[3] - acc[1].avg[3]) * 100
        pbar_str = "---Avg_Loss.0:{:>5.3f} Avg_Loss.1:{:>5.5f} D_Acc:{:>2.2f} " \
                   "Acc.0:[r:{:>5.2f} c:{:>5.2f} f:{:>5.2f} all:{:>5.2f}] Acc.1:[r:{:>5.2f} c:{:>5.2f} f:{:>5.2f} all:{:>5.2f}] Epoch_Time:{:>5.2f}min---".format(
            all_loss[0].avg, all_loss[1].avg, epoch_delta_acc,
            acc[0].avg[0] * 100, acc[0].avg[1] * 100,
            acc[0].avg[2] * 100, acc[0].avg[3] * 100,
            acc[1].avg[0] * 100, acc[1].avg[1] * 100,
            acc[1].avg[2] * 100, acc[1].avg[3] * 100,
            (end_time - start_time) / 60
        )

        logger.info(pbar_str)
        return acc[0].avg[3], [all_loss[0].avg, all_loss[1].avg]
    else:
        pbar_str = "---Avg_Loss:{:>5.3f}   Epoch_Accuracy:{:>5.2f}%   Epoch_Time:{:>5.2f}min---".format(
            all_loss.avg, acc.avg * 100, (end_time - start_time) / 60
        )
        logger.info(pbar_str)
        return acc.avg, all_loss.avg


def valid_model(
        dataLoader, epoch_number, model, cfg, criterion, logger, device, **kwargs
):
    model.eval()
    num_classes = dataLoader.dataset.get_num_classes()
    fusion_matrix = FusionMatrix(num_classes)
    start_time = time.time()

    with torch.no_grad():
        if cfg.CLASSIFIER.TYPE == 'LDA':
            all_loss = [AverageMeter(), AverageMeter()]
            acc = [AverageMeterList(4), AverageMeterList(4)]
        else:
            all_loss = AverageMeter()
            acc = AverageMeter()
        func = torch.nn.Softmax(dim=1)
        for i, (image, label, meta) in enumerate(dataLoader):
            image, label = image.to(device), label.to(device)
            feature = model(image, feature_flag=True)

            output = model(feature, classifier_flag=True)

            if isinstance(output, list):

                loss1 = criterion(output[0], label)
                loss2 = criterion(output[-1], label)

                now_result = torch.argmax(func(output[0]), 1)
                all_loss[0].update(loss1.data.item(), label.shape[0])
                now_acc, cnt = accuracy_shot(now_result, label, meta['shot_cate'])
                acc[0].update(now_acc, cnt)

                now_result = torch.argmax(func(output[1]), 1)
                all_loss[1].update(loss2.data.item(), label.shape[0])
                now_acc, cnt = accuracy_shot(now_result, label, meta['shot_cate'])
                acc[1].update(now_acc, cnt)

            else:
                loss = criterion(output, label)
                score_result = func(output)

                now_result = torch.argmax(score_result, 1)
                all_loss.update(loss.data.item(), label.shape[0])
                fusion_matrix.update(now_result.cpu().numpy(), label.cpu().numpy())
                now_acc, cnt = accuracy(now_result.cpu().numpy(), label.cpu().numpy())
                acc.update(now_acc, cnt)

        end_time = time.time()
        if cfg.CLASSIFIER.TYPE == 'LDA':
            pbar_str = "------- Valid: Epoch:{:>3d}  Valid_Loss.0:{:>5.3f} Valid_Loss.1:{:>5.3f} Acc.0:[r:{:>5.2f} c:{:>5.2f} f:{:>5.2f} all:{:>5.2f}] " \
                       "Acc.1:[r:{:>5.2f} c:{:>5.2f} f:{:>5.2f} all:{:>5.2f}] Time:{:>5.2f}min-------".format(
                epoch_number, all_loss[0].avg, all_loss[1].avg,
                acc[0].avg[0] * 100, acc[0].avg[1] * 100,
                acc[0].avg[2] * 100, acc[0].avg[3] * 100,
                acc[1].avg[0] * 100, acc[1].avg[1] * 100,
                acc[1].avg[2] * 100, acc[1].avg[3] * 100,
                (end_time - start_time) / 60
            )
        else:
            pbar_str = "------- Valid: Epoch:{:>3d}  Valid_Loss:{:>5.3f}   Valid_Acc:{:>5.2f}%  Time:{:>5.2f}min-------".format(
                epoch_number, all_loss.avg, acc.avg * 100, (end_time - start_time) / 60
            )
        logger.info(pbar_str)
    if cfg.CLASSIFIER.TYPE == 'LDA':
        if cfg.AUTO_ALPHA.LOSS1_FACTOR == 0:
            return acc[0].avg[3], all_loss[0].avg + all_loss[1].avg
        return acc[1].avg[3], all_loss[0].avg + all_loss[1].avg
    return acc.avg, all_loss.avg
