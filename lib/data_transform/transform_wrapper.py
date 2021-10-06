from utils.registry import Registry
import torchvision.transforms as transforms


TRANSFORMS = Registry()


@TRANSFORMS.register("random_resized_crop")
def random_resized_crop(cfg, **kwargs):
    size = kwargs["input_size"] if kwargs["input_size"] != None else cfg.INPUT_SIZE
    return transforms.RandomResizedCrop(
        size=size,
        scale=cfg.TRANSFORMS.PROCESS_DETAIL.RANDOM_RESIZED_CROP.SCALE,
        ratio=cfg.TRANSFORMS.PROCESS_DETAIL.RANDOM_RESIZED_CROP.RATIO,
    )


@TRANSFORMS.register("random_crop")
def random_crop(cfg, **kwargs):
    size = kwargs["input_size"] if kwargs["input_size"] != None else cfg.INPUT_SIZE
    return transforms.RandomCrop(
        size, padding=cfg.TRANSFORMS.PROCESS_DETAIL.RANDOM_CROP.PADDING
    )


@TRANSFORMS.register("random_horizontal_flip")
def random_horizontal_flip(cfg, **kwargs):
    return transforms.RandomHorizontalFlip(p=0.5)


@TRANSFORMS.register("shorter_resize_for_crop")
def shorter_resize_for_crop(cfg, **kwargs):
    size = kwargs["input_size"] if kwargs["input_size"] != None else cfg.INPUT_SIZE
    assert size[0] == size[1], "this img-process only process square-image"
    return transforms.Resize(int(size[0] / 0.875))


@TRANSFORMS.register("normal_resize")
def normal_resize(cfg, **kwargs):
    size = kwargs["input_size"] if kwargs["input_size"] != None else cfg.INPUT_SIZE
    return transforms.Resize(size)


@TRANSFORMS.register("center_crop")
def center_crop(cfg, **kwargs):
    size = kwargs["input_size"] if kwargs["input_size"] != None else cfg.INPUT_SIZE
    return transforms.CenterCrop(size)


@TRANSFORMS.register("ten_crop")
def ten_crop(cfg, **kwargs):
    size = kwargs["input_size"] if kwargs["input_size"] != None else cfg.INPUT_SIZE
    return transforms.TenCrop(size)


@TRANSFORMS.register("normalize")
def normalize(cfg, **kwargs):
    if cfg.DATASET.DATASET == 'iNaturalist':
        return transforms.Normalize(
            mean=[0.466, 0.471, 0.380], std=[0.195, 0.194, 0.192]
        )
    elif cfg.DATASET.DATASET in ['ImageNet', 'Places2']:
        return transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else: # cifar
        return transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))


@TRANSFORMS.register("color_jitter")
def color_jitter(cfg, **kwargs):
    return transforms.ColorJitter(brightness=cfg.TRANSFORMS.PROCESS_DETAIL.COLOR_JITTER.brightness,
                                  contrast=cfg.TRANSFORMS.PROCESS_DETAIL.COLOR_JITTER.contrast,
                                  saturation=cfg.TRANSFORMS.PROCESS_DETAIL.COLOR_JITTER.saturation,
                                  hue=cfg.TRANSFORMS.PROCESS_DETAIL.COLOR_JITTER.hue)