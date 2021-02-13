# Data Transform

you can add dataset support yourself like that and
 will be used in path 'data_processor/data_loader'.

#### ImageNet transform
```python
# train dataset example for image-net
def imagenet_train_dataset(train_dir, transform=ImageNetTrainTransform):
    return datasets.ImageFolder(train_dir, transform)


# val dataset example for image-net
def imagenet_val_dataset(val_dir, transform=ImageNetValidationTransform):
    return datasets.ImageFolder(val_dir, transform)


# test dataset example for image-net
def imagenet_test_dataset(testdir, transform=ImageNetTestTransform):
    return datasets.ImageFolder(testdir, transform)
```

# TinyImageNet Transform

```python
# train-dataset example for tiny-image-net
def classify_train_dataset(train_dir, transform=TinyImageNetTrainTransform):
    return datasets.ImageFolder(train_dir, transform)


# val-dataset example for tiny-image-net
def classify_val_dataset(val_dir, transform=TinyImageNetvalidationTransform):
    return datasets.ImageFolder(val_dir, transform)


# test-dataset example for tiny-image-net
def classify_test_dataset(testdir, transform=TinyImageNetTestTransform):
    return datasets.ImageFolder(testdir, transform)
```
