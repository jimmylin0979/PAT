from .resnet import build_resnet_backbone_kd
from .fpn import build_resnet_fpn_backbone_kd, build_mobilenetv2_fpn_backbone
from .swin_transformer import build_swint_fpn_backbone, build_retinanet_swint_fpn_backbone
from .fpn_wAFP import *

from .registry import register_new_forward
