import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone import res50, bbn_res50, res32_cifar, bbn_res32_cifar, res101, res10, res18, res152, resnext50, resnext101, resnext152
from modules import GAP, Identity, FCNorm, NonLinearNeck, LinearNeck

import pdb

class Network(nn.Module):
    def __init__(self, cfg, logger, mode="train", num_classes=1000):
        super(Network, self).__init__()
        self.logger = logger
        pretrain = (
            True
            if mode == "train"
            and cfg.RESUME_MODEL == ""
            and cfg.BACKBONE.PRETRAINED_MODEL != ""
            else False
        )

        self.num_classes = num_classes
        self.cfg = cfg

        self.backbone = eval(self.cfg.BACKBONE.TYPE)(
            self.cfg,
            pretrain=pretrain,
            pretrained_model=cfg.BACKBONE.PRETRAINED_MODEL,
            last_layer_stride=2,
        )
        self.module = self._get_module()
        
        if self.cfg.CLASSIFIER.NECK.ENABLE:
            self.neck = self._get_neck_module()
            
        self.classifier = self._get_classifer()
        self.feature_len = self.get_feature_length()
        
        self.i = 0


    def forward(self, x, **kwargs):
        if "feature_flag" in kwargs or "feature_cb" in kwargs or "feature_rb" in kwargs:
            return self.extract_feature(x, **kwargs)
        elif "classifier_flag" in kwargs:
            if isinstance(self.classifier, torch.nn.modules.container.Sequential):
                out = []
                out.append(self.classifier[0](x))
                out.append(self.classifier[-1](x))
                return out
            else:
                return self.classifier(x)

        x = self.backbone(x)
        x = self.module(x)
        x = x.view(x.shape[0], -1)

        if isinstance(self.classifier, torch.nn.modules.container.Sequential):
            out = []
            out.append(self.classifier[0](x))
            out.append(self.classifier[-1](x))
            
            if self.cfg.CLASSIFIER.NECK.ENABLE:
                x_f = self.neck(x)
                return out, x_f
            return out
        else:
            x = self.classifier(x)
        return x


    def extract_feature(self, x, **kwargs):
        if "bbn" in self.cfg.BACKBONE.TYPE:
            x = self.backbone(x, **kwargs)
        else:
            x = self.backbone(x)
        x = self.module(x)
        x = x.view(x.shape[0], -1)

        return x


    def freeze_backbone(self):
        self.logger.info("Freezing backbone .......")
        for p in self.backbone.parameters():
            p.requires_grad = False

    def refreeze_last_block(self):
        self.logger.info("Refreezing the last block of backbone......")
        for p in self.backbone.layer4[2].parameters():
            p.requires_grad = True

    def refreeze_last_stage(self):
        self.logger.info("Refreezing the last stage of backbone......")
        for p in self.backbone.layer4.parameters():
            p.requires_grad = True

    def load_backbone_model(self, backbone_path=""):
        self.backbone.load_model(backbone_path)
        self.logger.info("Backbone has been loaded...")


    def load_model(self, model_path):
        pretrain_dict = torch.load(
            model_path, map_location="cpu" if self.cfg.CPU_MODE else "cuda"
        )
        pretrain_dict = pretrain_dict['state_dict'] if 'state_dict' in pretrain_dict else pretrain_dict
        model_dict = self.state_dict()
        from collections import OrderedDict
        new_dict = OrderedDict()
        for k, v in pretrain_dict.items():
            if k.startswith("module"):
                new_dict[k[7:]] = v
            else:
                new_dict[k] = v

        model_dict.update(new_dict)
        self.load_state_dict(model_dict)
        self.logger.info("Model has been loaded...")


    def get_feature_length(self):
        if "cifar" in self.cfg.BACKBONE.TYPE:
            num_features = 64
        elif self.cfg.BACKBONE.TYPE in ['res10',]:
            num_features = 512
        else:
            num_features = 2048

        if "bbn" in self.cfg.BACKBONE.TYPE:
            num_features = num_features * 2
        return num_features


    def _get_module(self):
        module_type = self.cfg.MODULE.TYPE
        if module_type == "GAP":
            module = GAP()
        elif module_type == "Identity":
            module= Identity()
        else:
            raise NotImplementedError

        return module

    def _get_neck_module(self):
        if self.cfg.CLASSIFIER.NECK.TYPE == 'NonLinear':
            return NonLinearNeck(self.cfg.CLASSIFIER.NECK.NUM_FEATURES, self.cfg.CLASSIFIER.NECK.NUM_OUT, self.cfg.CLASSIFIER.NECK.HIDDEN_DIM)
        elif self.cfg.CLASSIFIER.NECK.TYPE == 'Identity':
            return Identity()
        else:
            return LinearNeck(self.cfg.CLASSIFIER.NECK.NUM_FEATURES, self.cfg.CLASSIFIER.NECK.NUM_OUT)

    def _get_classifer(self):
        bias_flag = self.cfg.CLASSIFIER.BIAS

        num_features = self.get_feature_length()
        if self.cfg.CLASSIFIER.TYPE == "FCNorm":
            classifier = FCNorm(num_features, self.num_classes)
        elif self.cfg.CLASSIFIER.TYPE == "FC":
            classifier = nn.Linear(num_features, self.num_classes, bias=bias_flag)
        elif self.cfg.CLASSIFIER.TYPE == 'LDA':
            classifier = []
            for i in range(2):
                classifier.append(nn.Linear(num_features, self.num_classes, bias=bias_flag))
            classifier = nn.Sequential(*classifier)
        else:
            raise NotImplementedError

        return classifier
