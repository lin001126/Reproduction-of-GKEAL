# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset, Subset
from torchvision.transforms import transforms
from itertools import chain, repeat
from random import Random
from typing import Callable, Iterable, List, Optional, TypeVar

T_co = TypeVar('T_co', covariant=True)

class DatasetWrapper(Dataset[T_co]):
    basic_transform: Callable[[T_co], T_co]
    augment_transform: Callable[[T_co], T_co]

    def __init__(
        self,
        labels: Iterable[int],
        base_ratio: float,
        num_phases: int,
        augment: bool,
        inplace_repeat: int = 1,
        shuffle_seed: Optional[int] = None,
        num_shots: int = 5,  
        augment_count: int = 100  
    ) -> None:
        # 类型提示
        self.dataset: Dataset[T_co]
        self.num_classes: int

        # 初始化
        super().__init__()
        self.inplace_repeat = inplace_repeat
        self.base_ratio = base_ratio
        self.num_phases = num_phases
        self.base_size = int(self.num_classes * self.base_ratio)
        self.incremental_size = self.num_classes - self.base_size
        self.phase_size = self.incremental_size // num_phases if num_phases > 0 else 0
        self.num_shots = num_shots
        self.augment_count = augment_count
        print(f"Initialized with num_shots: {num_shots}, augment_count: {augment_count}")
        # 创建每个类别的索引列表
        self.class_indices: List[List[int]] = [[] for _ in range(self.num_classes)]
        for idx, label in enumerate(labels):
            self.class_indices[label].append(idx)
        self._transform = self.augment_transform if augment else self.basic_transform

        self.real_labels: List[int] = list(range(self.num_classes))
        if shuffle_seed is not None:
            Random(shuffle_seed).shuffle(self.real_labels)
            Random(shuffle_seed).shuffle(self.class_indices)
        
        print(f"Initialized with num_shots: {num_shots}, augment_count: {augment_count}")  

    def __getitem__(self, index: int) -> T_co:
        return self._transform(self.dataset[index])

    def _subset(self, label_begin: int, label_end: int) -> Subset[T_co]:
        sub_ids = list(chain.from_iterable(self.class_indices[label_begin:label_end]))
        return Subset(self, list(chain.from_iterable(repeat(sub_ids, self.inplace_repeat))))

    def subset_at_phase(self, phase: int) -> Subset[T_co]:
        if phase == 0:
            return self._subset(0, self.base_size)
        return self._subset(
            self.base_size + (phase - 1) * self.phase_size,
            self.base_size + phase * self.phase_size,
        )

    def subset_until_phase(self, phase: int) -> Subset[T_co]:
        return self._subset(
            0,
            self.base_size + phase * self.phase_size,
        )

    def few_shot_sampler(self, phase: int, random_seed: Optional[int] = None) -> Subset[T_co]:
        few_shot_indices = []
        rng = Random(random_seed) if random_seed is not None else Random()
    
        # get subset
        subset = self._subset(self.base_size + (phase - 1) * self.phase_size, self.base_size + phase * self.phase_size)
    
        # few-shot sampling is performed following the subset of data in the current stage
        for class_idx in range(self.num_classes):
            indices = [i for i in subset.indices if self.dataset[i][1] == class_idx]  
            rng.shuffle(indices)  # shuffle
            if len(indices) > self.num_shots:
                few_shot_indices.extend(indices[:self.num_shots])
            else:
                few_shot_indices.extend(indices)
    
        print(f"Few-shot indices count: {len(few_shot_indices)}") 
        return Subset(self, few_shot_indices)


    def augment_data(self, x: torch.Tensor) -> torch.Tensor:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.RandomCrop(x.size()[-2:], padding=4),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        augmented_data = []
        for i in range(x.size(0)):
            for _ in range(self.augment_count):
                augmented_data.append(transform(x[i]))
        augmented_data_tensor = torch.stack(augmented_data)
        print(f"Original tensor shape: {x.shape}, Augmented data shape: {augmented_data_tensor.shape}")
        return augmented_data_tensor
