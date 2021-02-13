# Data Transform

you can add transform yourself like that.
These transform codes will be used in path 'data_processor/dataset' 
while program loading dataset from your disk.

#### ImageNet transform
```python
# iamgenet examples
ImageNetTrainTransform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    ImageNetNormalize,
])

ImageNetValidationTransform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    ImageNetNormalize,
])

ImageNetTestTransform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    ImageNetNormalize,
])
```

# TinyImageNet Transform

```python
# iamgenet examples
ImageNetTrainTransform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    ImageNetNormalize,
])

ImageNetValidationTransform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    ImageNetNormalize,
])

ImageNetTestTransform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    ImageNetNormalize,
])
```
