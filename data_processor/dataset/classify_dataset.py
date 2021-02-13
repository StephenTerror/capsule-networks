import os
from torchvision import datasets
from torchvision.transforms import transforms

from data_processor.data_transform import TinyImageNetTrainTransform
from data_processor.data_transform import TinyImageNetvalidationTransform
from data_processor.data_transform import TinyImageNetTestTransform

# Image Net support, may be use in future.
from data_processor.data_transform import ImageNetTrainTransform
from data_processor.data_transform import ImageNetValidationTransform
from data_processor.data_transform import ImageNetTestTransform

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


# That may be use in future, keep it.
def classify_dataset(data_dir, transform, not_strict=False):
    if not os.path.exists(data_dir) and not_strict:
        print("path ==> '%s' is not found" % data_dir)
        return

    return datasets.ImageFolder(data_dir, transform)


# train-dataset example for tiny-image-net
def classify_train_dataset(train_dir, transform=TinyImageNetTrainTransform):
    return datasets.ImageFolder(train_dir, transform)


# val-dataset example for tiny-image-net
def classify_val_dataset(val_dir, transform=TinyImageNetvalidationTransform):
    return datasets.ImageFolder(val_dir, transform)


# test-dataset example for tiny-image-net
def classify_test_dataset(testdir, transform=TinyImageNetTestTransform):
    return datasets.ImageFolder(testdir, transform)


# Image Net support, may be use in future.

# train dataset example for image-net
def imagenet_train_dataset(train_dir, transform=ImageNetTrainTransform):
    return datasets.ImageFolder(train_dir, transform)


# val dataset example for image-net
def imagenet_val_dataset(val_dir, transform=ImageNetValidationTransform):
    return datasets.ImageFolder(val_dir, transform)


# test dataset example for image-net
def imagenet_test_dataset(testdir, transform=ImageNetTestTransform):
    return datasets.ImageFolder(testdir, transform)
