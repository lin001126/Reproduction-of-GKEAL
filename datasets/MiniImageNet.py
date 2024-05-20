# -*- coding: utf-8 -*-
from typing import Tuple
import torch
from .DatasetWrapper import DatasetWrapper
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2 as transforms
from os import path
from torch import Tensor

class MiniImageNet(DatasetWrapper[Tuple[torch.Tensor, int]]):
    num_classes = 100
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
        split = 'train' if train else 'val'
        self.dataset = ImageFolder(path.join(root, split), transform=self.basic_transform if not augment else self.augment_transform)
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
        return img, target
