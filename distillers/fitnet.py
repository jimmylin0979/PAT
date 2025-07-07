import torch
import torch.nn.functional as F
import torch.nn as nn
import math

from timm.models.layers import PatchEmbed

from ._base import BaseDistiller
from .registry import register_distiller
from .utils import (
    get_module_dict,
    init_weights,
    is_cnn_model,
    kd_loss,
    LambdaModule,
    MyPatchMerging,
    set_module_dict,
    TokenFilter,
    Unpatchify,
)


@register_distiller
class FitNet(BaseDistiller):
    requires_feat = True

    def __init__(self, student, teacher, criterion, args, **kwargs):
        super(FitNet, self).__init__(student, teacher, criterion, args)
        self.student = student
        self.teacher = teacher
        self.criterion = criterion
        self.args = args
        self.ce_loss_weight = self.args.gt_loss_weight
        self.kd_loss_weight = self.args.kd_loss_weight

        self.projector = nn.ModuleDict()

        is_cnn_teacher = is_cnn_model(teacher)
        is_cnn_student = is_cnn_model(student)
        self.is_cnn_teacher = is_cnn_teacher
        self.is_cnn_student = is_cnn_student

        for stage in self.args.fitnet_stages:
            stage_modules = nn.ModuleDict()

            _, size_t = teacher.stage_info(stage)
            _, size_s = student.stage_info(stage)

            if is_cnn_student and is_cnn_teacher:
                feature_filter_s = nn.Identity()
                feature_filter_t = nn.Identity()
                aligner = nn.Conv2d(size_s[0], size_t[0], 1, 1, 0, bias=False)

            elif is_cnn_student and not is_cnn_teacher:
                token_num = getattr(teacher, "num_tokens", 0)  # cls tokens
                feature_filter_s = nn.Identity()
                in_chans, H, W = size_s
                patch_num, embed_dim = size_t
                patch_grid = int((patch_num - token_num) ** 0.5)
                if H >= patch_grid:
                    feature_filter_t = TokenFilter(token_num, remove_mode=True)  # remove cls token
                    patch_size = H // patch_grid
                    assert patch_size * patch_grid == H
                    aligner = PatchEmbed(H, patch_size, in_chans, embed_dim)
                else:
                    feature_filter_t = nn.Sequential(
                        TokenFilter(token_num, remove_mode=True), MyPatchMerging(H * W)  # remove cls token
                    )
                    scale = patch_grid // H
                    assert scale * H == patch_grid
                    aligner = nn.Sequential(
                        nn.Conv2d(in_chans, embed_dim * scale**2, 1, 1, 0),
                        LambdaModule(lambda x: torch.einsum("nchw->nhwc", x)),
                        nn.Flatten(start_dim=1, end_dim=2),
                    )

            elif not is_cnn_student and is_cnn_teacher:
                token_num = getattr(student, "num_tokens", 0)  # cls tokens
                patch_num, embed_dim = size_s
                in_chans, H, W = size_t
                patch_grid = int((patch_num - token_num) ** 0.5)
                feature_filter_s = TokenFilter(token_num, remove_mode=True)  # remove cls token
                feature_filter_t = nn.Identity()
                if H >= patch_grid:
                    patch_size = H // patch_grid
                    assert patch_size * patch_grid == H
                    aligner = nn.Sequential(nn.Linear(embed_dim, patch_size**2 * in_chans), Unpatchify(patch_size))
                else:
                    assert patch_grid % H == 0
                    scale = patch_grid // H
                    aligner = nn.Sequential(
                        MyPatchMerging(H**2),
                        LambdaModule(
                            lambda x: torch.einsum("nhwc->nchw", x.view(x.size(0), H, H, x.size(-1))).contiguous()
                        ),
                        nn.Conv2d(embed_dim * scale**2, in_chans, 1, 1, 0),
                    )

            else:
                token_num_s = getattr(student, "num_tokens", 0)  # cls tokens
                token_num_t = getattr(teacher, "num_tokens", 0)  # cls tokens
                patch_num_s, embed_dim_s = size_s
                patch_num_t, embed_dim_t = size_t
                patch_num_s -= token_num_s
                patch_num_t -= token_num_t
                feature_filter_s = TokenFilter(token_num_s, remove_mode=True)  # remove cls token
                feature_filter_t = TokenFilter(token_num_t, remove_mode=True)  # remove cls token
                if patch_num_t > patch_num_s:
                    scale = patch_num_t * embed_dim_t // (patch_num_s * embed_dim_s)
                    assert scale * patch_num_s * embed_dim_s == patch_num_t * embed_dim_t
                    aligner = nn.Sequential(
                        nn.Linear(embed_dim_s, embed_dim_s * scale),
                        LambdaModule(lambda x, embed_dim_t=embed_dim_t: x.reshape(shape=(x.size(0), -1, embed_dim_t))),
                    )
                else:
                    assert patch_num_s % patch_num_t == 0
                    in_feature = patch_num_s * embed_dim_s // patch_num_t
                    aligner = nn.Sequential(MyPatchMerging(patch_num_t), nn.Linear(in_feature, size_t[-1]))

            stage_modules["feature_filter_s"] = feature_filter_s
            stage_modules["feature_filter_t"] = feature_filter_t
            stage_modules["aligner"] = aligner

            set_module_dict(self.projector, stage, stage_modules)

        self.init()

    def init(self, verbose=True):
        self.projector.apply(init_weights)
        if verbose and self.args.rank == 0:
            print(self.args)
            print(f"additional params: {sum([p.numel() for p in self.projector.parameters()]) / 1e6:.2f}M")
            print(self.projector)

    def forward(self, image, label, epoch):
        with torch.no_grad():
            self.teacher.eval()
            logits_teacher, feat_teacher = self.teacher(image, requires_feat=True)

        logits_student, feat_student = self.student(image, requires_feat=True)

        fitnet_losses = []
        for stage in self.args.fitnet_stages:
            idx_t, _ = self.teacher.stage_info(stage)
            idx_s, _ = self.student.stage_info(stage)

            feat_t = feat_teacher[idx_t]
            feat_s = feat_student[idx_s]

            feat_t = get_module_dict(self.projector, stage)["feature_filter_t"](feat_t)
            feat_s = get_module_dict(self.projector, stage)["feature_filter_s"](feat_s)

            if not self.is_cnn_teacher and not math.sqrt(feat_t.shape[1]).is_integer():
                feat_t = feat_t[:, 1:, :]
            if not self.is_cnn_student and not math.sqrt(feat_s.shape[1]).is_integer():
                feat_s = feat_s[:, 1:, :]

            feat_s_aligned = get_module_dict(self.projector, stage)["aligner"](feat_s)

            fitnet_losses.append(F.mse_loss(feat_s_aligned, feat_t))

        loss_fitnet = self.args.fitnet_loss_weight * sum(fitnet_losses)

        loss_ce = self.args.gt_loss_weight * self.criterion(logits_student, label)
        # to make general fitnet the same as the one for conv, just set kd_loss_weight to 0
        loss_kd = self.args.kd_loss_weight * kd_loss(logits_student, logits_teacher, self.args.kd_temperature)
        losses_dict = {
            "loss_gt": loss_ce,
            "loss_kd": loss_kd,
            "loss_fitnet": loss_fitnet,
        }
        return logits_student, losses_dict
