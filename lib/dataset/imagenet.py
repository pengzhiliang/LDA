from torch.utils.data import Dataset
import torch
import numpy as np
import json, os, random, time
import cv2
import torchvision.transforms as transforms
from data_transform.transform_wrapper import TRANSFORMS
import pdb

class ImageNet(Dataset):
    cls_num = 1000
    def __init__(self, mode="train", cfg=None, transform=None):
        self.mode = mode
        self.transform = transform
        self.cfg = cfg
        self.input_size = cfg.INPUT_SIZE
        self.data_type = cfg.DATASET.DATA_TYPE
        self.color_space = cfg.COLOR_SPACE
        self.size = self.input_size
        self.dual_sample = True if cfg.TRAIN.SAMPLER.DUAL_SAMPLER.ENABLE and mode == "train" else False

        print("Use {} Mode to train network".format(self.color_space))
        if self.data_type != "nori":
            self.data_root = cfg.DATASET.ROOT
        else:
            self.fetcher = None

        if self.mode == "train":
            print("Loading train data ...", end=" ")
            self.json_path = cfg.DATASET.TRAIN_JSON
        elif "valid" in self.mode:
            print("Loading valid data ...", end=" ")
            self.json_path = cfg.DATASET.VALID_JSON
        else:
            raise NotImplementedError
        self.update_transform()

        self.data = []
        with open(self.json_path, "r") as f:
            lines = f.readlines()
            for l in lines:
                l_split = l.strip().split(" ")
                self.data.append([l_split[0], int(l_split[1])])

        self.num_classes = self.cls_num
        print("Contain {} images of {} classes".format(len(self.data), self.num_classes))
        
        if self.dual_sample or (self.cfg.TRAIN.SAMPLER.TYPE == "weighted sampler" and self.mode == "train"):
            self.class_weight, self.sum_weight = self.get_weight(self.data, self.cls_num)
            self.class_dict = self._get_class_dict()
            
        img_num_list = np.array(self.get_cls_num_list())
        
        class_cate_index = np.zeros(self.num_classes).astype(np.int) # few
        class_cate_index[img_num_list >= 20] = 1 # medium
        class_cate_index[img_num_list > 100] = 2 # many
        self.class_cate_index = class_cate_index
        
        if self.cfg.CLASSIFIER.TYPE == 'LDA' and self.mode == 'train':
            self.class_weight_for_lda = self.get_class_weight_for_lda(img_num_list)


    def get_weight(self, annotations, num_classes):
        num_list = [0] * num_classes
        cat_list = []
        for anno in annotations:
            category_id = anno[1]
            num_list[category_id] += 1
            cat_list.append(category_id)
        max_num = max(num_list)
        class_weight = [max_num / i for i in num_list]
        sum_weight = sum(class_weight)
        return class_weight, sum_weight
    
    def _get_class_dict(self):
        class_dict = dict()
        for i, anno in enumerate(self.data):
            cat_id = anno[1]
            if not cat_id in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)
        return class_dict

    def get_class_weight_for_lda(self, img_num_list):
        cls_num_list = np.array(img_num_list)
        return 1 / (self.cls_num * cls_num_list)

    def sample_class_index_by_weight(self):
        rand_number, now_sum = random.random() * self.sum_weight, 0
        for i in range(self.cls_num):
            now_sum += self.class_weight[i]
            if rand_number <= now_sum:
                return i
            
    def __getitem__(self, index):
        
        if self.cfg.TRAIN.SAMPLER.TYPE == "weighted sampler" and self.mode == "train":
            assert self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE in ["balance", "reverse"]
            if self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE == "balance":
                sample_class = random.randint(0, self.cls_num - 1)
                
            elif self.cfg.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE == "reverse":
                sample_class = self.sample_class_index_by_weight()
            sample_indexes = self.class_dict[sample_class]
            index = random.choice(sample_indexes)
                
        now_info = self.data[index]
        img = self._get_image(now_info)
        meta = dict()
        image = self.transform(img)
        image_label = (
            now_info[1] if "test" not in self.mode else 0
        )  # 0-index
        
        if self.dual_sample:
            if self.cfg.TRAIN.SAMPLER.DUAL_SAMPLER.TYPE == "reverse":
                sample_class = self.sample_class_index_by_weight()
                sample_indexes = self.class_dict[sample_class]
                sample_index = random.choice(sample_indexes)
            elif self.cfg.TRAIN.SAMPLER.DUAL_SAMPLER.TYPE == "balance":
                sample_class = random.randint(0, self.cls_num - 1)
                sample_indexes = self.class_dict[sample_class]
                sample_index = random.choice(sample_indexes)
            elif self.cfg.TRAIN.SAMPLER.DUAL_SAMPLER.TYPE == "uniform":
                sample_index = random.randint(0, self.__len__() - 1)

            sample_img, sample_label = self.data[sample_index], self.data[sample_index][1]
            sample_img = self._get_image(sample_img)
            sample_img = self.transform(sample_img)

            meta['sample_image'] = sample_img
            meta['sample_label'] = sample_label
            
        if self.mode not in ["train", "valid"]:
            meta["fpath"] = now_info[0]
        
        if self.cfg.CLASSIFIER.TYPE == 'LDA' and self.mode == 'train':
            meta['class_weight'] = self.class_weight_for_lda[image_label]
        meta['shot_cate'] = self.class_cate_index[image_label]
        
        return image, image_label, meta

    def update_transform(self, input_size=None):
        normalize = TRANSFORMS["normalize"](cfg=self.cfg, input_size=input_size)
        transform_list = [transforms.ToPILImage()]
        transform_ops = (
            self.cfg.TRANSFORMS.TRAIN_TRANSFORMS
            if self.mode == "train"
            else self.cfg.TRANSFORMS.TEST_TRANSFORMS
        )
        for tran in transform_ops:
            if tran == 'None':
                continue
            transform_list.append(TRANSFORMS[tran](cfg=self.cfg, input_size=input_size))
        transform_list.extend([transforms.ToTensor(), normalize])
        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.data)

    def imread_with_retry(self, fpath):
        retry_time = 10
        for k in range(retry_time):
            try:
                img = cv2.imread(fpath)
                if img is None:
                    print("img is None, try to re-read img")
                    continue
                return img
            except Exception as e:
                if k == retry_time - 1:
                    assert False, "cv2 imread {} failed".format(fpath)
                time.sleep(0.1)

    def _get_image(self, now_info):
        fpath = os.path.join(self.data_root, now_info[0])
        img = self.imread_with_retry(fpath)

        if self.color_space == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def reset_epoch(self, cur_epoch):
        self.epoch = cur_epoch

    def get_num_classes(self):
        return self.num_classes

    def get_annotations(self):
        return self.data

    def get_cls_num_list(self):
        cls_num_list = [0, ] * self.num_classes
        for d in self.data:
            cls_num_list[d[1]] += 1
        return cls_num_list