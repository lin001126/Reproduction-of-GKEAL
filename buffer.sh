#!/bin/bash


path="./backbones/resnet32_CIFAR-100_0.6_None"
model="resnet32"
C=10

python main.py GKEAL --dataset CIFAR-100 --base-ratio 0.6 --phases 8 \
    --data-root ~/dataset --IL-batch-size 4096 --num-workers 16 --backbone $model \
    --gamma 0.1 --sigma 10 --buffer-size 1000 \
    --backbone-path $path\
    --use-afc --augment-count $C
python main.py GKEAL --dataset CIFAR-100 --base-ratio 0.6 --phases 8 \
    --data-root ~/dataset --IL-batch-size 4096 --num-workers 16 --backbone $model \
    --gamma 0.1 --sigma 10 --buffer-size 2000 \
    --backbone-path $path\
    --use-afc --augment-count $C
python main.py GKEAL --dataset CIFAR-100 --base-ratio 0.6 --phases 8 \
    --data-root ~/dataset --IL-batch-size 4096 --num-workers 16 --backbone $model \
    --gamma 0.1 --sigma 10 --buffer-size 5000 \
    --backbone-path $path\
    --use-afc --augment-count $C
python main.py GKEAL --dataset CIFAR-100 --base-ratio 0.6 --phases 8 \
    --data-root ~/dataset --IL-batch-size 4096 --num-workers 16 --backbone $model \
    --gamma 0.1 --sigma 10 --buffer-size 8192 \
    --backbone-path $path\
    --use-afc --augment-count $C
python main.py GKEAL --dataset CIFAR-100 --base-ratio 0.6 --phases 8 \
    --data-root ~/dataset --IL-batch-size 4096 --num-workers 16 --backbone $model \
    --gamma 0.1 --sigma 10 --buffer-size 10000 \
    --backbone-path $path\
    --use-afc --augment-count $C
python main.py GKEAL --dataset CIFAR-100 --base-ratio 0.6 --phases 8 \
    --data-root ~/dataset --IL-batch-size 4096 --num-workers 16 --backbone $model \
    --gamma 0.1 --sigma 10 --buffer-size 12000 \
    -backbone-path $path\
    --use-afc --augment-count $C
python main.py GKEAL --dataset CIFAR-100 --base-ratio 0.6 --phases 8 \
    --data-root ~/dataset --IL-batch-size 4096 --num-workers 16 --backbone $model \
    --gamma 0.1 --sigma 0.1 --buffer-size 8192 \
    --backbone-path $path\
    --use-afc --augment-count $C
python main.py GKEAL --dataset CIFAR-100 --base-ratio 0.6 --phases 8 \
    --data-root ~/dataset --IL-batch-size 4096 --num-workers 16 --backbone $model \
    --gamma 0.1 --sigma 1 --buffer-size 8192 \
    --backbone-path $path\
    --use-afc --augment-count $C
python main.py GKEAL --dataset CIFAR-100 --base-ratio 0.6 --phases 8 \
    --data-root ~/dataset --IL-batch-size 4096 --num-workers 16 --backbone $model \
    --gamma 0.1 --sigma 2 --buffer-size 8192 \
    --backbone-path $path\
    --use-afc --augment-count $C
python main.py GKEAL --dataset CIFAR-100 --base-ratio 0.6 --phases 8 \
    --data-root ~/dataset --IL-batch-size 4096 --num-workers 16 --backbone $model \
    --gamma 0.1 --sigma 5 --buffer-size 8192 \
    --backbone-path $path\
    --use-afc --augment-count $C

python main.py GKEAL --dataset CIFAR-100 --base-ratio 0.6 --phases 8 \
    --data-root ~/dataset --IL-batch-size 4096 --num-workers 16 --backbone $model \
    --gamma 0.1 --sigma 10 --buffer-size 8192 \
    --backbone-path $path\
    --use-afc --augment-count $C
python main.py GKEAL --dataset CIFAR-100 --base-ratio 0.6 --phases 8 \
    --data-root ~/dataset --IL-batch-size 4096 --num-workers 16 --backbone $model \
    --gamma 0.1 --sigma 12 --buffer-size 8192 \
    --backbone-path $path\
    --use-afc --augment-count $C
python main.py GKEAL --dataset CIFAR-100 --base-ratio 0.6 --phases 8 \
    --data-root ~/dataset --IL-batch-size 4096 --num-workers 16 --backbone $model \
    --gamma 0.1 --sigma 15 --buffer-size 8192 \
    --backbone-path $path\
    --use-afc --augment-count $C
python main.py GKEAL --dataset CIFAR-100 --base-ratio 0.6 --phases 8 \
    --data-root ~/dataset --IL-batch-size 4096 --num-workers 16 --backbone $model \
    --gamma 0.1 --sigma 20 --buffer-size 8192 \
    --backbone-path $path\
    --use-afc --augment-count $C