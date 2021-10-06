# Long-tailed Distribution Adaptation (Accepted in ACM MM2021)

This project is built upon [BBN](https://github.com/Megvii-Nanjing/BBN).

## Installation
```bash
pip install -r requirements.txt
```

## Usage
1. Training
```bash
python main/train.py --cfg configs/ImageNet.yaml DATASET.ROOT /path/ImageNet_ILSVRC2012
```
2. Inference
```bash
python main/valid.py --cfg configs/ImageNet.yaml \
  TEST.MODEL_FILE ./output/LDA/ImageNet/LDA.ImageNet.resnext50.90epoch/models/best_model.pth \
  DATASET.VALID_JSON './datasets/ImageNet_LT/ImageNet_LT_test.txt'\
  DATASET.ROOT /path/ImageNet_ILSVRC2012
```