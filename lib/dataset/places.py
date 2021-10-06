from torch.utils.data import Dataset
import torch
import numpy as np
import json, os, random, time
import cv2
from .imagenet import ImageNet
import pdb

class Places2(ImageNet):
    cls_num = 365
    def __init__(self, mode="train", cfg=None, transform=None):
        super(Places2, self).__init__(mode, cfg, transform)

    def _get_image(self, now_info):
        # Train: train/airfield/00004851.jpg 0
        # Val: train/airfield/00002049.jpg 0
        # Test: val/airfield/Places365_val_00003597.jpg 0
        split_path = now_info[0].split('/')
        if '-' in split_path[1]:
            split_path[1] = split_path[1].replace('-', '/')
        if self.mode == 'train':
            fpath = os.path.join(self.data_root, 'train', split_path[1][0], split_path[1], split_path[-1])
        else:
            if split_path[0] == 'train':  # real validation stage
                fpath = os.path.join(self.data_root, 'train', split_path[1][0], split_path[1], split_path[-1])
            else: # test stage
                fpath = os.path.join(self.data_root, 'val', split_path[-1])
        # print(fpath)
        img = self.imread_with_retry(fpath)

        if self.color_space == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def imread_with_retry(self, fpath):
        retry_time = 10
        for k in range(retry_time):
            try:
                img = cv2.imread(fpath)
                if img is None:

                    print("img is None, try to re-read img, path is {}".format(fpath))
                    continue
                return img
            except Exception as e:
                if k == retry_time - 1:
                    assert False, "cv2 imread {} failed".format(fpath)
                time.sleep(0.1)