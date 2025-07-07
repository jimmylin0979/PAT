"""
Reference from https://github.com/megvii-research/mdistiller/blob/master/mdistiller/distillers/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import (
    init_weights,
    get_module_dict,
    set_module_dict,
    kd_loss,
)


def build_kd_trans(cfg):
    student_shapes = teacher_shapes = [
        [1, 256, 200, 304],  # [c, h, w]
        [1, 256, 100, 152],
        [1, 256, 50, 76],
        [1, 256, 25, 38],
        [1, 256, 13, 19],
    ]
    model = FitNet(student_shapes, teacher_shapes, cfg)
    return model


class ConvReg(nn.Module):
    """Convolutional regression"""

    def __init__(self, s_shape, t_shape, use_relu=True):
        super(ConvReg, self).__init__()
        self.use_relu = use_relu
        _, s_C, s_H, s_W = s_shape
        _, t_C, t_H, t_W = t_shape
        if s_H == 2 * t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=3, stride=2, padding=1)
        elif s_H * 2 == t_H:
            self.conv = nn.ConvTranspose2d(s_C, t_C, kernel_size=4, stride=2, padding=1)
        elif s_H >= t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=(1 + s_H - t_H, 1 + s_W - t_W))
        else:
            raise NotImplemented("student size {}, teacher size {}".format(s_H, t_H))
        self.bn = nn.BatchNorm2d(t_C)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.use_relu:
            return self.relu(self.bn(x))
        else:
            return self.bn(x)


class FitNet(nn.Module):
    """FitNets: Hints for Thin Deep Nets"""

    def __init__(self, feat_s_shapes, feat_t_shapes, cfg):
        super(FitNet, self).__init__()
        self.feat_loss_weight = cfg.KD.FITNET.FEAT_LOSS_WEIGHT
        self.kd_loss_weight = cfg.KD.FITNET.KD_LOSS_WEIGHT
        self.temperature = cfg.KD.FITNET.TEMPERATURE

        self.hint_layer = cfg.KD.FITNET.STAGE
        self.conv_reg = nn.ModuleList()
        for stage in self.hint_layer:
            module = ConvReg(feat_s_shapes[stage], feat_t_shapes[stage])
            self.conv_reg.append(module)

    def forward(self, pred_student, feat_student, pred_teacher, feat_teacher, **kwargs):

        #
        logits_student, _ = pred_student
        logits_teacher, _ = pred_teacher

        # losses
        loss_feat = 0
        for stage in self.hint_layer:
            f_s = self.conv_reg[stage](feat_student[stage])
            loss_feat += F.mse_loss(f_s, feat_teacher[stage])

        loss_feat = self.feat_loss_weight * loss_feat
        loss_kd = self.kd_loss_weight * kd_loss(logits_student, logits_teacher, self.temperature)
        losses_dict = {
            "loss_kd": loss_kd,
            "loss_feat": loss_feat,
        }
        return losses_dict
