# COCO object detection and instance segmentation

PS: based on the [DKD's codebase](https://github.com/megvii-research/mdistiller/tree/master/detection).

## Environment

* 4 GPUs
* python 3.10
* torch 1.13.0
* torchvision 0.14.0

## Installation

Our code is based on Detectron2, please install Detectron2 refer to https://github.com/facebookresearch/detectron2.

Please put the [COCO](https://cocodataset.org/#download) dataset in datasets/.

Please put the pretrained weights for teacher and student in pretrained/. You can find the pretrained weights [here](https://github.com/dvlab-research/ReviewKD/releases/). The pretrained models we provided contains both teacher's and student's weights. The teacher's weights come from Detectron2's pretrained detector. The student's weights are ImageNet pretrained weights.

## Roadmap

+ [ ] Upload pretrained weights, like `r18-swint.pth` for heterogeneous distillation for detection

## Training

```bash
# Tea: Swin-T, Stu: R-18
# DKD
python3 train_net.py --config-file configs/DKD/DKD-R18-SwinT.yaml --num-gpus 4

# ReviewKD
python3 train_net.py --config-file configs/ReviewKD/ReviewKD-R18-SwinT.yaml --num-gpus 4

# OFA
python3 train_net.py --config-file configs/OFA/OFA-R18-SwinT.yaml --num-gpus 4

# PAT
# set TEST.EVAL_PERIOD to 0 (steps) to disable the middle validation output
python3 train_net.py --config-file configs/PAT/PAT-R18-SwinT.yaml --num-gpus 4 --dist-url tcp://127.0.0.1:62950
```

## Acknowledge

+ This repository is built based on [SwinT_detectron2](https://github.com/xiaohu2015/SwinT_detectron2), [detectron2_ema](https://github.com/xiaohu2015/detectron2_ema/tree/main), [DKD](https://github.com/megvii-research/mdistiller) and the [Detectron2](https://github.com/facebookresearch/detectron2) library. Thanks for their help.
