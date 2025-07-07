### imagenet ###
###############################################################################
# ofa
# Deit -> ResNet18
python -m torch.distributed.launch --nproc_per_node=4 train.py /work/u9616567/imagenet1k/ILSVRC/Data/CLS-LOC --datset imagenet \
    --config configs/imagenet/cnn.yaml    --model resnet18\
    --teacher deit_tiny_patch16_224 --teacher-pretrained ./pretrained/imagenet_teachers/deit_tiny_patch16_224-a1311bcf.pth \
    --amp --model-ema --output ./output/imagenet \
    --distiller ofa --ofa-eps 1.5

###############################################################################
# off
### CNN-Based Student ###
# DeiT_T -> ResNet18
python -m torch.distributed.launch --nproc_per_node=4 --master_port=29600 train.py ./data/imagenet1k/ILSVRC/Data/CLS-LOC --dataset imagenet \
    --config configs/imagenet/cnn.yaml    --model resnet18 \
    --teacher deit_tiny_patch16_224 --teacher-pretrained ./pretrained/imagenet_teachers/deit_tiny_patch16_224-a1311bcf.pth \
    --amp --model-ema --pin-mem --output ./output/imagenet \
    --distiller pat

# Swin_T -> ResNet18
python -m torch.distributed.launch --nproc_per_node=4 --master_port=29600 train.py ./data/imagenet1k/ILSVRC/Data/CLS-LOC --dataset imagenet \
    --config configs/imagenet/cnn.yaml    --model resnet18 \
    --teacher swin_tiny_patch4_window7_224 --teacher-pretrained ./pretrained/imagenet_teachers/swin_tiny_patch4_window7_224.pth \
    --amp --model-ema --pin-mem --output ./output/imagenet \
    --distiller pat

# Mixer-B/16 -> ResNet18
python -m torch.distributed.launch --nproc_per_node=4 --master_port=29600 train.py ./data/imagenet1k/ILSVRC/Data/CLS-LOC --dataset imagenet \
    --config configs/imagenet/cnn.yaml    --model resnet18 \
    --teacher mixer_b16_224 --teacher-pretrained ./pretrained/imagenet_teachers/jx_mixer_b16_224-76587d61.pth \
    --amp --model-ema --pin-mem --output ./output/imagenet \
    --distiller pat

# DeiT_T -> MobileNetV2
python -m torch.distributed.launch --nproc_per_node=4 --master_port=29600 train.py ./data/imagenet1k/ILSVRC/Data/CLS-LOC --dataset imagenet \
    --config configs/imagenet/cnn.yaml    --model mobilenetv2_100 \
    --teacher deit_tiny_patch16_224 --teacher-pretrained ./pretrained/imagenet_teachers/deit_tiny_patch16_224-a1311bcf.pth \
    --amp --model-ema --pin-mem --output ./output/imagenet \
    --distiller pat

# Swin_T -> MobileNetV2
python -m torch.distributed.launch --nproc_per_node=4 --master_port=29600 train.py ./data/imagenet1k/ILSVRC/Data/CLS-LOC --dataset imagenet \
    --config configs/imagenet/cnn.yaml    --model mobilenetv2_100 \
    --teacher swin_tiny_patch4_window7_224 --teacher-pretrained ./pretrained/imagenet_teachers/swin_tiny_patch4_window7_224.pth \
    --amp --model-ema --pin-mem --output ./output/imagenet \
    --distiller pat

# Mixer-B/16 -> MobileNetV2
python -m torch.distributed.launch --nproc_per_node=4 --master_port=29600 train.py ./data/imagenet1k/ILSVRC/Data/CLS-LOC --dataset imagenet \
    --config configs/imagenet/cnn.yaml    --model mobilenetv2_100 \
    --teacher mixer_b16_224 --teacher-pretrained ./pretrained/imagenet_teachers/jx_mixer_b16_224-76587d61.pth \
    --amp --model-ema --pin-mem --output ./output/imagenet \
    --distiller pat


### ViT-Based Student ###
# ResNet50 -> DeiT_T
python -m torch.distributed.launch --nproc_per_node=4 --master_port=29600 train.py ./data/imagenet1k/ILSVRC/Data/CLS-LOC --dataset imagenet \
    --config configs/imagenet/vit_mlp.yaml    --model deit_tiny_patch16_224 \
    --teacher resnet50 --teacher-pretrained ./pretrained/imagenet_teachers/resnet50_a1_0-14fe96d1.pth \
    --amp --model-ema --pin-mem --output ./output/imagenet \
    --distiller pat

# ConvNeXt_T -> DeiT_T
python -m torch.distributed.launch --nproc_per_node=4 --master_port=29600 train.py ./data/imagenet1k/ILSVRC/Data/CLS-LOC --dataset imagenet \
    --config configs/imagenet/vit_mlp.yaml    --model deit_tiny_patch16_224 \
    --teacher convnext_tiny --teacher-pretrained ./pretrained/imagenet_teachers/convnext_tiny_1k_224_ema.pth \
    --amp --model-ema --pin-mem --output ./output/imagenet \
    --distiller pat

# Mixer-B/16 -> DeiT_T, Acc
python -m torch.distributed.launch --nproc_per_node=4 --master_port=29600 train.py ./data/imagenet1k/ILSVRC/Data/CLS-LOC --dataset imagenet \
    --config configs/imagenet/vit_mlp.yaml    --model deit_tiny_patch16_224 \
    --teacher mixer_b16_224 --teacher-pretrained ./pretrained/imagenet_teachers/jx_mixer_b16_224-76587d61.pth \
    --amp --model-ema --pin-mem --output ./output/imagenet \
    --distiller pat

# ResNet50 -> Swin_N
python -m torch.distributed.launch --nproc_per_node=4 --master_port=29600 train.py ./data/imagenet1k/ILSVRC/Data/CLS-LOC --dataset imagenet \
    --config configs/imagenet/vit_mlp.yaml    --model swin_nano_patch4_window7_224 \
    --teacher resnet50 --teacher-pretrained ./pretrained/imagenet_teachers/resnet50_a1_0-14fe96d1.pth \
    --amp --model-ema --pin-mem --output ./output/imagenet \
    --distiller pat

# ConvNeXt_T -> Swin_N
python -m torch.distributed.launch --nproc_per_node=4 --master_port=29600 train.py ./data/imagenet1k/ILSVRC/Data/CLS-LOC --dataset imagenet \
    --config configs/imagenet/vit_mlp.yaml    --model swin_nano_patch4_window7_224 \
    --teacher convnext_tiny --teacher-pretrained ./pretrained/imagenet_teachers/convnext_tiny_1k_224_ema.pth \
    --amp --model-ema --pin-mem --output ./output/imagenet \
    --distiller pat

# Mixer-B/16 -> Swin_N
python -m torch.distributed.launch --nproc_per_node=4 --master_port=29600 train.py ./data/imagenet1k/ILSVRC/Data/CLS-LOC --dataset imagenet \
    --config configs/imagenet/vit_mlp.yaml    --model swin_nano_patch4_window7_224 \
    --teacher mixer_b16_224 --teacher-pretrained ./pretrained/imagenet_teachers/jx_mixer_b16_224-76587d61.pth \
    --amp --model-ema --pin-mem --output ./output/imagenet \
    --distiller pat

### MLP-Based Student ###
# ResNet50 -> ResMLP-S12
python -m torch.distributed.launch --nproc_per_node=4 --master_port=29600 train.py ./data/imagenet1k/ILSVRC/Data/CLS-LOC --dataset imagenet \
    --config configs/imagenet/vit_mlp.yaml    --model resmlp_12_224 \
    --teacher resnet50 --teacher-pretrained ./pretrained/imagenet_teachers/resnet50_a1_0-14fe96d1.pth \
    --amp --model-ema --pin-mem --output ./output/imagenet \
    --distiller pat

# ConvNeXt_T -> ResMLP-S12
python -m torch.distributed.launch --nproc_per_node=4 --master_port=29600 train.py ./data/imagenet1k/ILSVRC/Data/CLS-LOC --dataset imagenet \
    --config configs/imagenet/vit_mlp.yaml    --model resmlp_12_224 \
    --teacher convnext_tiny --teacher-pretrained ./pretrained/imagenet_teachers/convnext_tiny_1k_224_ema.pth \
    --amp --model-ema --pin-mem --output ./output/imagenet \
    --distiller pat

# Swin_T -> ResMLP-S12
python -m torch.distributed.launch --nproc_per_node=4 --master_port=29600 train.py ./data/imagenet1k/ILSVRC/Data/CLS-LOC --dataset imagenet \
    --config configs/imagenet/vit_mlp.yaml    --model resmlp_12_224 \
    --teacher swin_tiny_patch4_window7_224 --teacher-pretrained ./pretrained/imagenet_teachers/swin_tiny_patch4_window7_224.pth \
    --amp --model-ema --pin-mem --output ./output/imagenet \
    --distiller pat
