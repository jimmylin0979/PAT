import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import Block
from .utils import (
    GAP1d,
    get_module_dict,
    init_weights,
    is_cnn_model,
    PatchMerging,
    SepConv,
    set_module_dict,
    TokenFilter,
    TokenFnContext,
)


def build_kd_trans(cfg):
    #
    is_cnn_student = is_cnn_teacher = True
    feat_s_shapes = feat_t_shapes = [
        [256, 320, 336],  # [c, h, w]
        [256, 160, 168],
        [256, 80, 84],
        [256, 40, 42],
        [256, 20, 21],
    ]
    model = OFA(is_cnn_student, feat_s_shapes, is_cnn_teacher, feat_t_shapes, cfg)
    return model


def ofa_loss(logits_student, logits_teacher, target_mask, eps, temperature=1.0):
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    prod = (pred_teacher + target_mask) ** eps
    loss = torch.sum(-(prod - target_mask) * torch.log(pred_student), dim=-1)
    return loss.mean()


class OFA(nn.Module):
    def __init__(self, is_cnn_student, feat_s_shapes, is_cnn_teacher, feat_t_shapes, cfg, **kwargs):
        super(OFA, self).__init__()

        # hyper-parameters
        self.ofa_eps = cfg.KD.OFA.EPS
        self.ofa_stage = cfg.KD.OFA.STAGE
        self.ofa_loss_weight = cfg.KD.OFA.OFA_LOSS_WEIGHT
        self.ofa_temperature = cfg.KD.OFA.TEMPERATURE
        self.kd_loss_weight = cfg.KD.OFA.KD_LOSS_WEIGHT
        self.num_classes = cfg.TEACHER.MODEL.ROI_HEADS.NUM_CLASSES + 1  # one extra class for background

        #
        # if len(self.ofa_eps) == 1:
        #     eps = [self.ofa_eps[0] for _ in range(len(self.ofa_stage) + 1)]
        #     self.ofa_eps = eps

        # assert len(self.ofa_stage) + 1 == len(self.ofa_eps)  # +1 for logits

        self.projector = nn.ModuleDict()

        feature_dim_t = feat_t_shapes[-1][1]
        feature_dim_s = feat_s_shapes[-1][1]
        box_feature_dim = 1024
        self.projector = nn.Sequential(
            nn.Linear(box_feature_dim, box_feature_dim),
            nn.BatchNorm1d(box_feature_dim),
            nn.Linear(box_feature_dim, self.num_classes),
        )
        self.projector.apply(init_weights)

    def get_learnable_parameters(self):
        student_params = 0
        extra_params = 0
        for n, p in self.named_parameters():
            if n.startswith("student"):
                student_params += p.numel()
            elif n.startswith("teacher"):
                continue
            else:
                if p.requires_grad:
                    extra_params += p.numel()
        return student_params, extra_params

    def forward(
        self, student_predictions, student_box_features, teacher_predictions, feat_teacher, label, *args, **kwargs
    ):

        # student_predictions   : tuple(torch.Tensor[total_proposals, 80+1], torch.Tensor[total_proposals, 4x80])
        # student_box_features  : List[torch.Tensor[ci, hi, wi]] x 5 # 5 stages
        # teacher_predictions   : same as logits_student
        # feat_teacher          : List[torch.Tensor[ci, hi, wi]] x 5 # 5 stages
        # label                 : List[torch.Tensor(num_proposals)] x bs

        stu_logits, stu_bbox_offsets = student_predictions
        tea_logits, tea_bbox_offsets = teacher_predictions

        num_classes = stu_logits.size(-1)
        gt_classes = torch.cat(tuple(label), 0).reshape(-1)
        target_mask = F.one_hot(gt_classes, num_classes)

        # Proposal are allocated among all stages, and return in the shape
        # for stage, eps in zip(self.ofa_stage, self.ofa_eps):
        logits_student_head = self.projector(student_box_features)  # [total_proposals, 1024] -> [total_proposals, 81]
        ofa_losses = ofa_loss(logits_student_head, tea_logits, target_mask, self.ofa_eps, self.ofa_temperature)
        loss_ofa = self.ofa_loss_weight * ofa_losses

        # ground-truth loss is already included with detectron2
        # loss_gt = self.args.gt_loss_weight * self.criterion(logits_student, label)
        # FIXME: cause Nan
        # loss_kd = self.kd_loss_weight * ofa_loss(
        #     stu_logits,
        #     tea_logits,
        #     target_mask,
        #     self.ofa_eps,  # TODO: Seperate the ofa_eps for intermidiate box_feature and final logit
        #     self.ofa_temperature,
        # )
        losses_dict = {"loss_ofa": loss_ofa}
        return losses_dict
