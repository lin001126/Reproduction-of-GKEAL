# -*- coding: utf-8 -*-
from typing import Tuple
import torch
from .DatasetWrapper import DatasetWrapper
from torchvision.datasets import ImageNet
from torchvision.transforms import v2 as transforms
from os import path
from torch import Tensor

class ImageNet_(DatasetWrapper[Tuple[torch.Tensor, int]]):
    num_classes = 1000
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    basic_transform = transforms.Compose(
        [
            transforms.Resize(232),
            transforms.CenterCrop(224),
            transforms.PILToTensor(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean, std, inplace=True),
            transforms.ToPureTensor(),
        ]
    )

    augment_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(176),
            transforms.RandomHorizontalFlip(0.5),
            transforms.TrivialAugmentWide(
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean, std, inplace=True),
            transforms.RandomErasing(0.1),
            transforms.ToPureTensor(),
        ]
    )

    def __init__(
        self,
        root: str,
        train: bool,
        base_ratio: float,
        num_phases: int,
        augment: bool = False,
        inplace_repeat: int = 1,
        shuffle_seed: int | None = None,
        num_shots: int = 5,  
        augment_count: int = 200
    ) -> None:
        root = path.expanduser(root)
        self.dataset = ImageNet(root, split="train" if train else "val")
        super().__init__(
            self.dataset.targets,
            base_ratio,
            num_phases,
            augment,
            inplace_repeat,
            shuffle_seed,
        )
    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        img, target = self.dataset[index]
        img = self._transform(img)
        return img, target

class ImageNetDog_(DatasetWrapper[Tuple[torch.Tensor, int]]):
    num_classes = 120
    mean = (0.4765, 0.4523, 0.3923)
    std = (0.2267, 0.2217, 0.2200)

    basic_transform = transforms.Compose(
        [
            transforms.Resize(232),
            transforms.CenterCrop(224),
            transforms.PILToTensor(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean, std, inplace=True),
            transforms.ToPureTensor(),
        ]
    )

    augment_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(176),
            transforms.RandomHorizontalFlip(0.5),
            transforms.TrivialAugmentWide(
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean, std, inplace=True),
            transforms.RandomErasing(0.1),
            transforms.ToPureTensor(),
        ]
    )

    def __init__(
        self,
        root: str,
        train: bool,
        base_ratio: float,
        num_phases: int,
        augment: bool = False,
        inplace_repeat: int = 1,
        shuffle_seed: int | None = None,
    ) -> None:
        root = "/hpc2hdd/home/ychen151/Analytic-continual-learning/imagenet-dog/images"
        split = "train" if train else "val"
        dataset_path = path.join(root, split)
        self.dataset = ImageFolder(dataset_path)
        super().__init__(
            self.dataset.targets,
            base_ratio,
            num_phases,
            augment,
            inplace_repeat,
            shuffle_seed,
        )