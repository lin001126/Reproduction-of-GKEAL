# Reproduction of GKEAL: Gaussian Kernel Embedded Analytic Learning for Few-shot Class Incremental Task

This project builds upon the [Analytic Continual Learning](https://github.com/ZHUANGHP/Analytic-continual-learning) repository by adding two new parameters: `--use-afc` and `--augment-count`. These parameters are designed to control the fewshot learning process. This work replicates the experiments from the paper "GKEAL: Gaussian Kernel Embedded Analytic Learning for Few-shot Class Incremental Task".

We primarily added the fewshot sampler and feature consolidation code in the main file and datasetwrapper, while also introducing new models and datasets in the dataset and model files. Additionally, we experimented with new backbones `Efficientnet` and `Vision Transformer`, as well as a new dataset `Imagenetdog`. The trained backbones of `ViT`, `ResNet`, and `EfficientNet` can be found [here](https://drive.google.com/file/d/1LCrX_Gz-AodRoRSbxWNX9o3w5Asc5BXV/view?usp=sharing).


For more detailed information on how to run the code and train the backbone from scratch, please refer to the [Analytic Continual Learning](https://github.com/ZHUANGHP/Analytic-continual-learning).

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/ZHUANGHP/Analytic-continual-learning.git
   cd Analytic-continual-learning
2. Install the required dependencies

## New Parameters
- `--use-afc`: This flag activates the use of Adaptive Feature Consolidation (AFC).
- `--augment-count`: Specifies the number of augmentations to be used in the fewshot learning process.

## Example Usage

To run the model with the new parameters, use the following command:

```bash
python main.py GKEAL --dataset CIFAR-100 --base-ratio 0.6 --phases 8 \
    --data-root ~/dataset --IL-batch-size 2560 --num-workers 16 --backbone resnet32 \
    --gamma 0.1 --sigma 10 --buffer-size 8192 \
    --backbone-path ./backbones/resnet32_CIFAR-100_0.6_None \
    --use-afc --augment-count 200
```bash

If you only want to perform continual learning, remove the use-afc and augment-count parameters:
```bash
python main.py GKEAL --dataset CIFAR-100 --base-ratio 0.6 --phases 8 \
    --data-root ~/dataset --IL-batch-size 2560 --num-workers 16 --backbone resnet32 \
    --gamma 0.1 --sigma 10 --buffer-size 8192 \
    --backbone-path ./backbones/resnet32_CIFAR-100_0.6_None


For more detailed information on how to run the code and train the backbone from scratch, please refer to the [Analytic Continual Learning](https://github.com/ZHUANGHP/Analytic-continual-learning).

