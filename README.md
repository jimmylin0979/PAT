# Perspective-AwareTeaching: Adapting Knowledge for Heterogeneous Distillation (PAT):

Official PyTorch implementation of **PAT**, from the following paper: \
Perspective-AwareTeaching: Adapting Knowledge for Heterogeneous Distillation

## Installation
First, clone the repository locally:

```bash
git clone https://github.com/jimmylin0979/PAT.git
```

Then, install PyTorch and [timm 0.6.5](https://github.com/huggingface/pytorch-image-models/tree/v0.6.5)

```bash
conda install -c pytorch pytorch torchvision
pip install timm==0.6.5
```

Our results are produced with `torch==1.13.0 torchvision==0.14.0 timm==0.6.5`. Other versions might also work. You can also construct the environment with docker container [Pytorch 22.08](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_22-08.html).

## Experiments

## Usage
### Data preparation

Download and extract ImageNet train and val images from http://image-net.org/. The directory structure is:

```
│path/to/imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

### Training

For CIFAR-100, Please put the pretrained weights for teacher to `pretrained/cifar_teachers`. You can find the pretrained weights [here](https://github.com/Hao840/OFAKD/releases/tag/checkpoint-cifar100). For ImageNet, please download from the timm manually and then store in `pretrained/imagenet_teachers`.

To train a resnet18 student using [DeiT-T teacher](https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth) on ImageNet on a single node with 8 GPUs, run:

```
python train.py ./data --dataset cifar100 --num-classes 100 \
    --config configs/cifar/vit_mlp.yaml    --model deit_tiny_patch16_224 \
    --teacher convnext_tiny --teacher-pretrained ./pretrained/cifar_teachers/convnext_tiny_cifar100.pth \
    --amp --model-ema --pin-mem --output ./output/cifar \
    --distiller pat
```

Other results can be reproduced following similar commands by modifying:

`--config `: configuration of training strategy.

`--model`: student model architecture.

`--teacher`: teacher model architecture.

`--teacher-pretrained`: path to checkpoint of pretrained teacher model.

`--distiller`: which KD algorithm to use.

For information about other tunable parameters, please refer to `train.py`.
For the detailed training commands, please refer to `command_cifar.sh` and `command_imagenet.sh`.

## Custom usage

**KD algorithm**: create new KD algorithm following examples in the `./distillers` folder.

**Model architecture**: create new model architecture following examples in the `./custom_model` folder. If intermediate features of the new model are required for KD, rewrite its `forward()` method following examples in the `./custom_forward` folder.

## Citation

```
```

## Pre-commit Installation

```
pip install pre-commit
pre-commit install --install-hooks
```

## Acknowledgement

+ This repository is built based on [OFA-KD](https://github.com/Hao840/OFAKD), [timm](https://github.com/rwightman/pytorch-image-models) and the [mdistiller](https://github.com/megvii-research/mdistiller) library. Thanks for their help.
+
