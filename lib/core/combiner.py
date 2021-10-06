import numpy as np
import torch, math
import torch.nn as nn
import torch.nn.functional as F
from core.evaluate import accuracy, accuracy_shot, accuracy_shot_perclass
import pdb
from loss import CrossEntropy


class Combiner:
    def __init__(self, cfg, device, **kwargs):
        self.cfg = cfg
        self.device = device
        self.epoch_number = cfg.TRAIN.MAX_EPOCH
        self.func = torch.nn.Softmax(dim=1)
        self.i = 0
        if self.cfg.AUTO_ALPHA.LENGTH <= 0:
            self.min_batch = max(1, int(sum(kwargs['num_class_list']) / (min(kwargs['num_class_list']) * cfg.TRAIN.BATCH_SIZE)))
        else:
            self.min_batch = self.cfg.AUTO_ALPHA.LENGTH

        if cfg.AUTO_ALPHA.ALPHA > 0:
            self.adap_alpha = Adaptive_Alpha(cfg, freeze=True, freeze_alpha=self.cfg.DATASET.ALPHA, length=self.min_batch, **kwargs)
        else:
            self.adap_alpha = Adaptive_Alpha(cfg, length=self.min_batch, **kwargs)

        self.CrossEntropy = CrossEntropy()


    def forward(self, model, criterion, image, label, meta, **kwargs):
        image, label = image.to(self.device), label.to(self.device)
        if 'class_weight' in meta:
            sample_weight = meta['class_weight'].to(image.device).float()

        if self.cfg.CLASSIFIER.NECK.ENABLE:
            output, x_f = model(image)

            all_features = x_f
            all_labels = label
            all_weights = sample_weight

            class_mean_features = []
            class_feature_num = []
            class_feature_weight = []
            class_intra_distance = []
            for l in all_labels.unique():
                label_index = all_labels == l
                center_feature = all_features[label_index].mean(0)
                class_mean_features.append(center_feature)
                if self.cfg.CLASSIFIER.NECK.INTRA_DISTANCE:
                    class_intra_distance.append((1 - torch.nn.functional.cosine_similarity(center_feature[None, :],
                                                                all_features[label_index])).mean())
                class_feature_num.append(label_index.sum())
                class_feature_weight.append(all_weights[label_index][0])

            class_mean_features = torch.stack(class_mean_features)
            class_feature_num = torch.stack(class_feature_num)
            class_feature_weight = torch.stack(class_feature_weight)
            c_batch = class_mean_features.shape[0]
            if self.cfg.CLASSIFIER.NECK.INTRA_DISTANCE:
                class_intra_distance = torch.stack(class_intra_distance)
            ignore_mask = 1 - torch.eye(c_batch).to(class_mean_features.device)
            distance_loss = 0 * class_mean_features.sum()

            if self.cfg.CLASSIFIER.NECK.INTER_DISTANCE:
                class_mean_features_norm = class_mean_features / class_mean_features.norm(dim=1)[:, None]
                distance_matrix = (1 - torch.mm(class_mean_features_norm, class_mean_features_norm.t()))
                class_inter_loss = torch.nn.functional.hinge_embedding_loss(distance_matrix,
                                -1 * torch.ones_like(distance_matrix).long(), margin=self.cfg.CLASSIFIER.NECK.MARGIN,
                                                                            reduction='none')

                if self.cfg.CLASSIFIER.NECK.WEIGHT_INTER_LOSS:
                    weight_norm_matrix = class_feature_weight[:, None].repeat(1, c_batch) + class_feature_weight[None, :].repeat(c_batch, 1)
                    weight_norm_matrix /= (weight_norm_matrix * ignore_mask).sum()
                    class_inter_loss = class_inter_loss * weight_norm_matrix

                if self.cfg.CLASSIFIER.NECK.WEIGHT_INTER_LOSS:
                    reduction_factor = 1
                else:
                    reduction_factor = all_labels.unique().shape[0] ** 2 - all_labels.unique().shape[0]

                class_inter_loss = (class_inter_loss * ignore_mask).sum() / reduction_factor
                distance_loss += class_inter_loss

            if self.cfg.CLASSIFIER.NECK.INTRA_DISTANCE:
                if (class_feature_num > 1).any():
                    class_intra_loss = class_intra_distance[class_feature_num > 1].mean()
                else:
                    class_intra_loss = (class_intra_distance * 0.).sum()
                distance_loss += class_intra_loss

            if self.i % self.cfg.SHOW_STEP == 0:
                kwargs['logger'].info(
                    "distance loss: {:1.6f}, inter: {:1.6f}, intra: {:1.6f}, Num[max:{}, mean:{:1.1f}], Distance[min:{:1.4f}, max:{:1.4f} mean: {:1.4f}]".format(
                        distance_loss * self.cfg.CLASSIFIER.NECK.LOSS_FACTOR,
                        0. if not self.cfg.CLASSIFIER.NECK.INTER_DISTANCE else class_inter_loss.item(),
                        0. if not self.cfg.CLASSIFIER.NECK.INTRA_DISTANCE else class_intra_loss.item(),
                        class_feature_num.max().item(), class_feature_num.float().mean().item(),
                        0. if not self.cfg.CLASSIFIER.NECK.INTER_DISTANCE else distance_matrix[
                            distance_matrix * ignore_mask > 0].min().item(),
                        0. if not self.cfg.CLASSIFIER.NECK.INTER_DISTANCE else distance_matrix.max().item(),
                        0. if not self.cfg.CLASSIFIER.NECK.INTER_DISTANCE else distance_matrix[
                            distance_matrix * ignore_mask > 0].mean().item()))

            self.i += 1

        else:
            output = model(image)

        if isinstance(output, list):
            loss = []
            loss_0 = criterion(output[0], label)

            now_acc = []
            now_cnt = []
            for o in output:
                now_result = torch.argmax(self.func(o), 1)
                acc, cnt = accuracy_shot(now_result, label, meta['shot_cate'])
                now_acc.append(acc)
                now_cnt.append(cnt)

            loss_1 = (sample_weight * criterion(output[-1], label, reduction='none')).sum()
            loss.append(loss_0 * self.cfg.AUTO_ALPHA.LOSS0_FACTOR)

            self.adap_alpha.update(now_acc[0][-1] * 100, now_acc[-1][-1] * 100)

            loss_1 = self.adap_alpha(loss_1)
            loss.append(loss_1 * self.cfg.AUTO_ALPHA.LOSS1_FACTOR)
            if self.cfg.CLASSIFIER.NECK.ENABLE:
                loss.append(distance_loss * self.cfg.CLASSIFIER.NECK.LOSS_FACTOR)

        else:
            loss = criterion(output, label)

            now_result = torch.argmax(self.func(output), 1)
            now_acc, now_cnt = accuracy(now_result.cpu().numpy(), label.cpu().numpy())

        return (loss, now_acc, now_cnt)


class Adaptive_Alpha(nn.Module):
    def __init__(self, cfg, freeze=False, eps=1e-6, freeze_alpha=1.0, **kwargs):
        super(Adaptive_Alpha, self).__init__()
        self.cfg = cfg
        self.freeze = freeze
        if freeze:
            self.alpha = torch.tensor(freeze_alpha).float().cuda()

        self.length = kwargs['length']  # cfg.AUTO_ALPHA.LENGTH
        self.acc0 = torch.zeros(self.length, dtype=torch.float).cuda()
        self.acc1 = torch.zeros(self.length, dtype=torch.float).cuda()
        # print("The length to calculate acc is {}".format(self.length))

        self.gamma = cfg.AUTO_ALPHA.GAMMA
        self.eps = eps
        self.n = 0
        self.scale_factor = 1

    def forward(self, x):
        return self.get_act_alpha() * x

    def update(self, acc0, acc1):

        self.acc0[self.n % self.length] = acc0
        self.acc1[self.n % self.length] = acc1

        self.n += 1

    def update_scale_factor(self, scale_factor):
        self.scale_factor = scale_factor

    @property
    def get_acc_diff(self):
        return self.acc0.mean() - self.acc1.mean()

    @property
    def get_acc_ratio(self):
        return self.acc1.mean() / self.acc0.mean()

    def get_act_alpha(self):
        if self.freeze:
            return self.alpha / self.scale_factor

        return max(self.get_acc_diff * 0., self.get_acc_diff) ** self.gamma
