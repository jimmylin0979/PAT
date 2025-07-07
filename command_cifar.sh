### cifar100 ###
###############################################################################
# ofa
# Swin_T -> ResNet18
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29600 train.py ./data --dataset cifar100 --num-classes 100 \
    --config configs/cifar/cnn.yaml --model resnet18 \
    --teacher swin_tiny_patch4_window7_224 --teacher-pretrained ./pretrained/cifar_teachers/swin_tiny_patch4_window7_224_cifar100.pth \
    --amp --model-ema --output ./output/cifar \
    --distiller ofa

# Swin_T -> ResMLP-S12, Acc
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29600 train.py ./data --dataset cifar100 --num-classes 100 \
    --config configs/cifar/vit_mlp.yaml --model resmlp_12_224 \
    --teacher swin_tiny_patch4_window7_224 --teacher-pretrained ./pretrained/cifar_teachers/swin_tiny_patch4_window7_224_cifar100.pth \
    --amp --model-ema --output ./output/cifar \
    --distiller ofa

###############################################################################
# fitnet
# Swin_T -> ResNet18, Acc 80.49
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29600 train.py ./data --dataset cifar100 --num-classes 100 \
    --config configs/cifar/cnn.yaml    --model resnet18\
    --teacher swin_tiny_patch4_window7_224 --teacher-pretrained ./pretrained/cifar_teachers/swin_tiny_patch4_window7_224_cifar100.pth \
    --amp --model-ema --output ./output/cifar \
    --distiller fitnet

###############################################################################
# off
### CNN-Based Student ###
# Swin_T -> ResNet18
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29600 train.py ./data --dataset cifar100 --num-classes 100 \
    --config configs/cifar/cnn.yaml    --model resnet18 \
    --teacher swin_tiny_patch4_window7_224 --teacher-pretrained ./pretrained/cifar_teachers/swin_tiny_patch4_window7_224_cifar100.pth \
    --amp --model-ema --pin-mem --output ./output/cifar \
    --distiller pat

# ViT_S -> ResNet18
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29600 train.py ./data --dataset cifar100 --num-classes 100 \
    --config configs/cifar/cnn.yaml    --model resnet18 \
    --teacher vit_small_patch16_224 --teacher-pretrained ./pretrained/cifar_teachers/vit_small_patch16_224_cifar100.pth \
    --amp --model-ema --pin-mem --output ./output/cifar \
    --distiller pat

# Mixer-B/16 -> ResNet18
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29600 train.py ./data --dataset cifar100 --num-classes 100 \
    --config configs/cifar/cnn.yaml    --model resnet18 \
    --teacher mixer_b16_224 --teacher-pretrained ./pretrained/cifar_teachers/mixer_b16_224_cifar100.pth \
    --amp --model-ema --pin-mem --output ./output/cifar \
    --distiller pat

# Swin_T -> MobileNetV2
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29600 train.py ./data --dataset cifar100 --num-classes 100 \
    --config configs/cifar/cnn.yaml    --model mobilenetv2_100 \
    --teacher swin_tiny_patch4_window7_224 --teacher-pretrained ./pretrained/cifar_teachers/swin_tiny_patch4_window7_224_cifar100.pth \
    --amp --model-ema --pin-mem --output ./output/cifar \
    --distiller pat

# ViT_S -> MobileNetV2
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29600 train.py ./data --dataset cifar100 --num-classes 100 \
    --config configs/cifar/cnn.yaml    --model mobilenetv2_100 \
    --teacher vit_small_patch16_224 --teacher-pretrained ./pretrained/cifar_teachers/vit_small_patch16_224_cifar100.pth \
    --amp --model-ema --pin-mem --output ./output/cifar \
    --distiller pat

# Mixer-B/16 -> MobileNetV2
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29600 train.py ./data --dataset cifar100 --num-classes 100 \
    --config configs/cifar/cnn.yaml    --model mobilenetv2_100 \
    --teacher mixer_b16_224 --teacher-pretrained ./pretrained/cifar_teachers/mixer_b16_224_cifar100.pth \
    --amp --model-ema --pin-mem --output ./output/cifar \
    --distiller pat


### ViT-Based Student ###
# ConvNeXt_T -> DeiT_T
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29600 train.py ./data --dataset cifar100 --num-classes 100 \
    --config configs/cifar/vit_mlp.yaml    --model deit_tiny_patch16_224 \
    --teacher convnext_tiny --teacher-pretrained ./pretrained/cifar_teachers/convnext_tiny_cifar100.pth \
    --amp --model-ema --pin-mem --output ./output/cifar \
    --distiller pat

# Mixer-B/16 -> DeiT_T
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29600 train.py ./data --dataset cifar100 --num-classes 100 \
    --config configs/cifar/vit_mlp.yaml    --model deit_tiny_patch16_224 \
    --teacher mixer_b16_224 --teacher-pretrained ./pretrained/cifar_teachers/mixer_b16_224_cifar100.pth \
    --amp --model-ema --pin-mem --output ./output/cifar \
    --distiller pat

# ConvNeXt_T -> Swin_P
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29600 train.py ./data --dataset cifar100 --num-classes 100 \
    --config configs/cifar/vit_mlp.yaml    --model swin_pico_patch4_window7_224 \
    --teacher convnext_tiny --teacher-pretrained ./pretrained/cifar_teachers/convnext_tiny_cifar100.pth \
    --amp --model-ema --pin-mem --output ./output/cifar \
    --distiller pat

# Mixer-B/16 -> Swin_P
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29600 train.py ./data --dataset cifar100 --num-classes 100 \
    --config configs/cifar/vit_mlp.yaml    --model swin_pico_patch4_window7_224 \
    --teacher mixer_b16_224 --teacher-pretrained ./pretrained/cifar_teachers/mixer_b16_224_cifar100.pth \
    --amp --model-ema --pin-mem --output ./output/cifar \
    --distiller pat

### MLP-Based Student ###
# ConvNeXt_T -> ResMLP-S12
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29600 train.py ./data --dataset cifar100 --num-classes 100 \
    --config configs/cifar/vit_mlp.yaml    --model resmlp_12_224 \
    --teacher convnext_tiny --teacher-pretrained ./pretrained/cifar_teachers/convnext_tiny_cifar100.pth \
    --amp --model-ema --pin-mem --output ./output/cifar \
    --distiller pat

# Swin_T -> ResMLP-S12
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29600 train.py ./data --dataset cifar100 --num-classes 100 \
    --config configs/cifar/vit_mlp.yaml    --model resmlp_12_224 \
    --teacher swin_tiny_patch4_window7_224 --teacher-pretrained ./pretrained/cifar_teachers/swin_tiny_patch4_window7_224_cifar100.pth \
    --amp --model-ema --pin-mem --output ./output/cifar \
    --distiller pat
