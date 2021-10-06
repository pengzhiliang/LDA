import _init_paths
from net import Network
from config import cfg, update_config
from dataset import *
import numpy as np
import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from core.evaluate import FusionMatrix, AverageMeterList, accuracy_shot
from utils.utils import create_valid_logger
from matplotlib import pyplot as plt

import pdb


def parse_args():
    parser = argparse.ArgumentParser(description="BBN evaluation")

    parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        required=True,
        default="configs/cifar10.yaml",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    return args


def valid_model(dataLoader, model, cfg, device, num_classes):
    result_list = []
    pbar = tqdm(total=len(dataLoader))
    model.eval()
    top1_count, top2_count, top3_count, index, fusion_matrix = (
        [],
        [],
        [],
        0,
        FusionMatrix(num_classes),
    )

    func = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        for i, (image, image_labels, meta) in enumerate(dataLoader):
            image = image.to(device)
            output = model(image)

            if isinstance(output, list):
                if cfg.CLASSIFIER.TYPE == 'LDA':
                    output = output[1]
                else:
                    output = torch.cat([o[:, 1:] for o in output], dim=1)
            result = func(output)
            _, top_k = result.topk(5, 1, True, True)
            score_result = result.cpu().numpy()
            fusion_matrix.update(score_result.argmax(axis=1), image_labels.numpy())
            topk_result = top_k.cpu().tolist()
            if not "image_id" in meta:
                meta["image_id"] = [0] * image.shape[0]
            image_ids = meta["image_id"]
            for i, image_id in enumerate(image_ids):
                result_list.append(
                    {
                        "image_id": image_id,
                        "image_label": int(image_labels[i]),
                        "top_3": topk_result[i],
                    }
                )
                top1_count += [topk_result[i][0] == image_labels[i]]
                top2_count += [image_labels[i] in topk_result[i][0:2]]
                top3_count += [image_labels[i] in topk_result[i][0:3]]
                index += 1
            now_acc = np.sum(top1_count) / index
            pbar.set_description("Now Top1:{:>5.2f}%".format(now_acc * 100))
            pbar.update(1)
    top1_acc = float(np.sum(top1_count) / len(top1_count))
    top2_acc = float(np.sum(top2_count) / len(top1_count))
    top3_acc = float(np.sum(top3_count) / len(top1_count))
    print(
        "Top1:{:>5.2f}%  Top2:{:>5.2f}%  Top3:{:>5.2f}%".format(
            top1_acc * 100, top2_acc * 100, top3_acc * 100
        )
    )
    pbar.close()

def valid_model_shot(
        dataLoader, model, cfg, device, num_classes, logger
):
    model.eval()

    with torch.no_grad():
        if cfg.CLASSIFIER.TYPE == 'LDA':
            acc = [AverageMeterList(4), AverageMeterList(4)]
        else:
            acc = AverageMeterList(4)
        func = torch.nn.Softmax(dim=1)
        for image, label, meta in tqdm(dataLoader):
            image, label = image.to(device), label.to(device)
            feature = model(image, feature_flag=True)

            output = model(feature, classifier_flag=True)

            if isinstance(output, list):
                now_result = torch.argmax(func(output[0]), 1)
                now_acc, cnt = accuracy_shot(now_result, label, meta['shot_cate'])
                acc[0].update(now_acc, cnt)

                now_result = torch.argmax(func(output[1]), 1)
                now_acc, cnt = accuracy_shot(now_result, label, meta['shot_cate'])
                acc[1].update(now_acc, cnt)

            else:
                score_result = func(output)

                now_result = torch.argmax(score_result, 1)
                now_acc, cnt = accuracy_shot(now_result, label, meta['shot_cate'])
                acc.update(now_acc, cnt)

        if cfg.CLASSIFIER.TYPE == 'LDA':
            pbar_str = "Test: Acc.0:[r:{:>5.2f} c:{:>5.2f} f:{:>5.2f} all:{:>5.2f}] Acc.1:[r:{:>5.2f} c:{:>5.2f} f:{:>5.2f} all:{:>5.2f}]".format(
                acc[0].avg[0] * 100, acc[0].avg[1] * 100, acc[0].avg[2] * 100, acc[0].avg[3] * 100, acc[1].avg[0] * 100,
                acc[1].avg[1] * 100, acc[1].avg[2] * 100, acc[1].avg[3] * 100
            )
        else:
            pbar_str = "Test: Acc:[r:{:>5.2f} c:{:>5.2f} f:{:>5.2f} all:{:>5.2f}]".format(acc.avg[0] * 100,
                                                                                          acc.avg[1] * 100,
                                                                                          acc.avg[2] * 100,
                                                                                          acc.avg[3] * 100)
        logger.info(pbar_str)


if __name__ == "__main__":
    args = parse_args()
    update_config(cfg, args)
    logger, log_file = create_valid_logger(cfg)

    test_set = eval(cfg.DATASET.DATASET)("valid", cfg)
    test_set.class_cate_index = eval(cfg.DATASET.DATASET)("train", cfg).class_cate_index

    num_classes = test_set.get_num_classes()
    device = torch.device("cpu" if cfg.CPU_MODE else "cuda")
    model = Network(cfg, mode="test", logger=logger, num_classes=num_classes)

    model_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "models")
    model_file = cfg.TEST.MODEL_FILE
    if "/" in model_file:
        model_path = model_file
    else:
        model_path = os.path.join(model_dir, model_file)
    model.load_model(model_path)

    if cfg.CPU_MODE:
        model = model.to(device)
    else:
        model = torch.nn.DataParallel(model).cuda()

    testLoader = DataLoader(
        test_set,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.TEST.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )
    logger.info('Test for getting shot accuracy')
    valid_model_shot(testLoader, model, cfg, device, num_classes, logger)
    # logger.info('Test for getting top-k accuracy')
    # valid_model(testLoader, model, cfg, device, num_classes)
