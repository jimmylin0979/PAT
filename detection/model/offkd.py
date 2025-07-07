import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import Block
from timm.models.layers import PatchEmbed
from timm.models.layers import _assert, trunc_normal_

from typing import List, Tuple

from .utils import (
    init_weights,
    is_cnn_model,
    get_module_dict,
    set_module_dict,
    TokenFilter,
    TokenFnContext,
    PatchMerging,
    SepConv,
    get_module_parameters,
    CudaCKA,
    kd_loss,
)


def build_kd_trans(cfg):

    # TODO: Whether detection needs to clarify the student, teacher model type
    is_cnn_student = True
    is_cnn_teacher = False
    teacher_shapes = [
        [1, 256, 256, 256],
        [1, 256, 128, 128],
        [1, 256, 64, 64],
        [1, 256, 32, 32],
        [1, 256, 16, 16],
    ]
    model = OFF(is_cnn_student, teacher_shapes, is_cnn_teacher, teacher_shapes, cfg)
    return model


def hcl_loss(feat_s, feat_t):
    loss_all = 0.0
    for fs, ft in zip(feat_s, feat_t):
        n, c, h, w = fs.shape
        loss = F.mse_loss(fs, ft, reduction="mean")
        cnt = 1.0
        tot = 1.0
        for l in [4, 2, 1]:
            if l >= h:
                continue
            tmpfs = F.adaptive_avg_pool2d(fs, (l, l))
            tmpft = F.adaptive_avg_pool2d(ft, (l, l))
            cnt /= 2.0
            loss += F.mse_loss(tmpfs, tmpft, reduction="mean") * cnt
            tot += cnt
        loss = loss / tot
        loss_all = loss_all + loss
    return loss_all


def mse_loss(feat_s: torch.Tensor, feat_t: torch.Tensor):
    loss_all = 0.0
    for fs, ft in zip(feat_s, feat_t):
        b, l, d = fs.shape
        loss = F.mse_loss(fs, ft, reduction="mean")
        loss_all = loss_all + loss
    return loss_all


def prior_loss(pl_feat_s: torch.Tensor, pl_feat_t: torch.Tensor):
    """
    Measure the piror removal in static by calculating the MSE loss between 2 CKA matrixes.

    Args:
        pl_feat_s (torch.Tensor): _description_
        pl_feat_t (torch.Tensor): _description_

    Returns:
        _type_: _description_
    """

    cka = CudaCKA(device=pl_feat_s.device)

    # pl_feat_s [b, num_stages, l, d]
    b, ns, l, d = pl_feat_s.shape
    loss_all = 0.0

    # version 2: vectorized
    # [b, ns, l, d] -> [ns, b, l * d]
    pl_feat_s = torch.reshape(pl_feat_s, (b, ns, -1))
    pl_feat_s = pl_feat_s.permute(1, 0, 2)
    pl_feat_t = torch.reshape(pl_feat_t, (b, ns, -1))
    pl_feat_t = pl_feat_t.permute(1, 0, 2)
    plow_cka = cka.linear_CKA(pl_feat_s, pl_feat_t)
    loss_all += torch.sum(1 - torch.abs(plow_cka))
    loss_all /= ns
    return loss_all


def reconstruction_loss(feat_orig: torch.Tensor, feat_rcon: torch.Tensor) -> torch.Tensor:
    """
    Calculates the mean squared error (MSE) loss between two feature tensors.

    Args:
        feat_orig (torch.Tensor): The original feature tensor.
        feat_rcon (torch.Tensor): The reconstructed feature tensor.

    Returns:
        torch.Tensor: The mean squared error (MSE) loss between feat_orig and feat_rcon.
    """
    loss = F.mse_loss(feat_rcon, feat_orig, reduction="mean")
    return loss


class OFF(nn.Module):
    def __init__(
        self,
        is_cnn_student: bool,
        student_shapes: List[torch.Size],
        is_cnn_teacher: bool,
        teacher_shapes: List[torch.Size],
        cfg,
        **kwargs,
    ):
        super().__init__()

        self.projector = nn.ModuleDict()

        print("student_shapes", student_shapes)
        print("teacher_shapes", teacher_shapes)

        self.is_cnn_teacher = is_cnn_teacher
        self.is_cnn_student = is_cnn_student
        self.student_shapes = student_shapes
        self.teacher_shapes = teacher_shapes

        #
        self.off_pl_loss_weight = cfg.OFFKD.PL_LOSS_WEIGHT
        self.off_pc_loss_weight = cfg.OFFKD.PC_LOSS_WEIGHT
        self.off_rc_loss_weight = cfg.OFFKD.RC_LOSS_WEIGHT
        self.off_embed_shape = cfg.OFFKD.EMBED_SHAPE
        self.off_afd = cfg.OFFKD.AFD
        self.off_afd_dim = cfg.OFFKD.AFD_DIM

        # FIXME: how to choose embed shape
        # choose a embed shape that is lower than both model
        # for deit -> res18, cls_token, 197 = 196 + 1
        # TODO: may choose embed_shape with the minium one between teacher and student
        # self.off_embed_shape = min(student_shapes[0][1], teacher_shapes[0][1])
        # embed_shape = torch.Size((1, 196, 256))
        embed_shape = torch.Size(self.off_embed_shape)
        self.projector = PLFE(is_cnn_student, student_shapes, is_cnn_teacher, teacher_shapes, embed_shape)

        # Attention-based Feature Distillation
        self.distiller = None
        if self.off_afd:
            dim = self.off_afd_dim
            num_stages = len(student_shapes)
            self.distiller = AFD(embed_shape, num_stages=num_stages, dim=dim)
            self.distiller.apply(init_weights)
            print("[MODEL] AFD.paras: ", get_module_parameters(self.distiller))

    def forward(self, feat_student, feat_teacher, *args, **kwargs):

        # to make trainging more general, reshape the student & teacher feature to a pre-defined dimension first
        for i in range(len(feat_teacher)):
            feat_teacher[i] = torch.nn.functional.interpolate(
                feat_teacher[i], size=self.teacher_shapes[i][-1], mode="nearest"
            )
            feat_student[i] = torch.nn.functional.interpolate(
                feat_student[i], size=self.teacher_shapes[i][-1], mode="nearest"
            )

        # implementation of OFF losses
        off_pl_feats_s, off_rc_feats_s = [], []
        off_pl_feats_t, off_rc_feats_t = [], []
        off_pl_losses = []
        off_rc_losses_s, off_rc_losses_t = [], []
        for stage in range(len(feat_teacher)):

            # remove cls_token before feeding into plfe encoder
            # print(idx_s, len(feat_student), len(feat_teacher))
            feat_s = feat_student[stage]
            feat_t = feat_teacher[stage]

            # No need to consider the feature dimension in detection, they are all 4 dimension feature
            # # how will token filtre do when dealing weith models like swin_tiny that do not have cls_token
            # # version 1
            # if not self.is_cnn_teacher and not math.sqrt(feat_t.shape[1]).is_integer():
            #     feat_t = feat_t[:, 1:, :]
            # if not self.is_cnn_student and not math.sqrt(feat_s.shape[1]).is_integer():
            #     feat_s = feat_s[:, 1:, :]
            # # # version 2
            # # if self.feat_filter_t is not None and not math.sqrt(feat_t.shape[1]).is_integer():
            # #     feat_t = self.feat_filter_t(feat_t)
            # # if self.feat_filter_s is not None and not math.sqrt(feat_s.shape[1]).is_integer():
            # #     feat_s = self.feat_filter_s(feat_s)

            # pl stanfs for prior-low, rc stands for reconstructed
            pl_feat_s, pl_feat_t, rc_feat_s, rc_feat_t = self.projector(feat_s, feat_t)
            off_pl_feats_s.append(pl_feat_s)
            off_pl_feats_t.append(pl_feat_t)
            # off_rc_feats_s.append(rc_feat_s)
            # off_rc_feats_t.append(rc_feat_t)
            # off_pl_losses.append(mse_loss([pl_feat_s], [pl_feat_t]))

            #
            # print("[DEBUG] pl_feat_t", pl_feat_t.shape)
            # print("[DEBUG] pl_feat_s", pl_feat_s.shape)
            # print("[DEBUG] rc_feat_t", rc_feat_t.shape, feat_t.shape)
            # print("[DEBUG] rc_feat_s", rc_feat_s.shape, feat_s.shape)
            off_rc_losses_t.append(reconstruction_loss(rc_feat_t, feat_t))
            off_rc_losses_s.append(reconstruction_loss(rc_feat_s, feat_s))

        # losses
        # prior-low mse loss
        off_pl_feats_s_st = torch.stack(off_pl_feats_s, dim=1)  # [b, n (num_stages), l, d]
        off_pl_feats_t_st = torch.stack(off_pl_feats_t, dim=1)
        if self.distiller is not None:
            loss_off_pl = self.distiller(off_pl_feats_s_st, off_pl_feats_t_st)
        else:
            loss_off_pl = F.mse_loss(off_pl_feats_s_st, off_pl_feats_t_st, reduction="mean")
        loss_off_pl *= self.off_pl_loss_weight  # * min((epoch + 1) / self.args.off_warmup_epochs, 1.0)

        # prior-content loss
        loss_off_pc = self.off_pc_loss_weight * prior_loss(off_pl_feats_s_st, off_pl_feats_t_st)

        # reconstruction loss
        loss_off_rc_s = self.off_rc_loss_weight * sum(off_rc_losses_s)
        loss_off_rc_t = self.off_rc_loss_weight * sum(off_rc_losses_t)

        # loss_gt = self.args.gt_loss_weight * self.criterion(logits_student, label)
        # loss_kd = self.args.kd_loss_weight * kd_loss(logits_student, logits_teacher, self.args.kd_temperature)

        losses_dict = {
            # "loss_gt": loss_gt,
            # "loss_kd": loss_kd,
            "loss_off_rc_s": loss_off_rc_s,
            "loss_off_rc_t": loss_off_rc_t,
            "loss_off_pc": loss_off_pc,
            "loss_off_pl": loss_off_pl,
        }
        return losses_dict


class PLFE(nn.Module):  #
    """Prior-Low Feature Extractor"""

    def __init__(
        self,
        is_cnn_student: bool,
        student_shapes: torch.Size,
        is_cnn_teacher: bool,
        teacher_shapes: torch.Size,
        embed_shape: torch.Size,
        *args,
        **kwargs,
    ):
        super(PLFE, self).__init__()

        self.is_cnn_teacher = is_cnn_teacher
        self.is_cnn_student = is_cnn_student

        # prior-low encoder
        num_stages = 1
        # for stage in range(len(student_shapes)):
        #     if not self.is_cnn_student:
        #         b, c, h, w = student_shapes[stage]
        #         student_shapes[stage] = torch.Size((b, c, h * w))
        #     if not self.is_cnn_teacher:
        #         b, c, h, w = teacher_shapes[stage]
        #         teacher_shapes[stage] = torch.Size((b, c, h * w))
        self.encoder = PLFEEncoder(student_shapes, teacher_shapes, embed_shape, num_stages)

        # prior-introduced decoder
        self.teacher_prior_decoder = PLFEDecoder(self.is_cnn_teacher, student_shapes, embed_shape, num_stages)
        self.student_prior_decoder = PLFEDecoder(self.is_cnn_student, teacher_shapes, embed_shape, num_stages)

        print("[MODEL] PLFEEncoder.paras: ", get_module_parameters(self.encoder))
        print("[MODEL] PLFEDecoder.paras (teacher-prior): ", get_module_parameters(self.teacher_prior_decoder))
        print("[MODEL] PLFEDecoder.paras (student-prior): ", get_module_parameters(self.student_prior_decoder))

    def forward(self, feat_student: torch.Tensor, feat_teacher: torch.Tensor):

        prior_low_feat_teacher = self.encoder(feat_teacher)
        prior_low_feat_student = self.encoder(feat_student)

        # print("[DEBUG] prior_low_feat_student", prior_low_feat_student.shape)
        # print("[DEBUG] prior_low_feat_teacher", prior_low_feat_teacher.shape)

        # reconstruct the features with different prior decoder
        reconstructed_feat_teacher = self.student_prior_decoder(prior_low_feat_teacher, feat_teacher.shape)
        reconstructed_feat_student = self.teacher_prior_decoder(prior_low_feat_student, feat_student.shape)

        return (
            prior_low_feat_student,
            prior_low_feat_teacher,
            reconstructed_feat_student,
            reconstructed_feat_teacher,
        )


class CNN2ViTProjector(nn.Module):
    def __init__(self, in_shapes, embed_tokens: int, embed_dim: int) -> None:
        super(CNN2ViTProjector, self).__init__()
        _, c, h, w = in_shapes
        img_size = h
        assert h == w, "feature width should be as same as the height"

        self.h = int(embed_tokens**0.5)
        # when patch size if smaller than 1, do interpolation first, or when ... # TODO
        patch_size = int(math.ceil(h / (embed_tokens**0.5)))
        # if patch_size < 1:
        #     patch_size = max(patch_size, 1)
        #     img_size = self.h
        # assert (
        #     patch_size >= 1
        # ), f"patch size should be at least 1, but found height = {h} and stride = {int(embed_tokens ** 0.5)}"

        # Reference Why do not use norm in PatchEmbed, https://github.com/facebookresearch/dino/issues/33
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(
            img_size=int(self.h * self.patch_size),
            patch_size=patch_size,
            in_chans=c,
            embed_dim=embed_dim,
        )

    def forward(self, x: torch.Tensor):
        # x.shape (b, c, h, w) -> (b, l, d)
        # print(x.shape, self.do_interpolation, self.h)
        if x.shape[2] < self.h:
            x = torch.nn.functional.interpolate(x, size=self.h, mode="nearest")
        elif x.shape[2] < int(self.patch_size * self.h):
            x = torch.nn.functional.interpolate(x, size=int(self.patch_size * self.h), mode="nearest")
        embed = self.patch_embed(x)
        return embed


class ViT2ViTProjector(nn.Module):
    def __init__(self, in_shapes, embed_tokens: int, embed_dim: int) -> None:
        super(ViT2ViTProjector, self).__init__()
        _, self.l1, self.d1 = in_shapes
        # TODO: use PatchMerging for downsampling is good, but how about upsampling ?
        self.mlp_d = nn.Sequential(
            nn.BatchNorm1d(self.l1),
            nn.Linear(self.d1, embed_dim),
        )
        self.mlp_l = nn.Sequential(
            nn.BatchNorm1d(embed_dim),
            nn.Linear(self.l1, embed_tokens),
        )

    def forward(self, x: torch.Tensor):
        # x.shape (b, l1, d1) -> (b, l, d)

        # (b, l1, d1) -> (b, l1, d)
        embed = self.mlp_d(x)
        # (b, l1, d) -> (b, d, l1)
        embed = embed.transpose(1, 2)
        # (b, d, l1) -> (b, d, l)
        embed = self.mlp_l(embed)
        # (b, d, l) -> (b, l, d)
        embed = embed.transpose(1, 2)
        return embed


class ViT2CNNProjector(nn.Module):
    def __init__(self, in_shapes: torch.Size, out_shapes: torch.Size) -> None:
        super(ViT2CNNProjector, self).__init__()

        _, l, d = in_shapes
        _, c, h, w = out_shapes
        assert h == w, "feature width should be as same as the height"

        self.h = h
        self.conv = nn.Sequential(
            # using depthwise convolution to save vram usage
            nn.BatchNorm2d(d),
            nn.Conv2d(d, c, kernel_size=1, stride=1, bias=False),
            # nn.Conv2d(d, c, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, embed: torch.Tensor):
        # embed.shape (b, l, d) -> (b, c, h, w)
        b, l, d = embed.shape
        h = w = int(math.sqrt(l))
        embed = embed.reshape(b, h, w, d)
        embed = embed.permute(0, 3, 1, 2)  # (b, d, h, w)
        embed = torch.nn.functional.interpolate(embed, size=self.h, mode="nearest")  # (b, d, h, w)
        embed = self.conv(embed)  # (b, c, h, w)
        return embed


class CNN2CNNProjector(nn.Module):
    def __init__(self, in_shapes: torch.Size, out_shapes: torch.Size) -> None:
        super(CNN2CNNProjector, self).__init__()

        _, ci, hi, wi = in_shapes
        _, co, ho, wo = out_shapes
        assert hi == wi, "feature width should be as same as the height"
        assert ho == wo, "feature width should be as same as the height"

        self.h = ho
        self.conv = nn.Sequential(
            nn.BatchNorm2d(ci),
            nn.Conv2d(ci, co, kernel_size=1, stride=1, bias=False),
            # nn.Conv2d(co, co, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, embed: torch.Tensor):
        # embed.shape (b, l, d) -> (b, c, h, w)
        embed = torch.nn.functional.interpolate(embed, size=self.h, mode="nearest")
        embed = self.conv(embed)  # (b, c, h, w)
        return embed


def get_projector_module(in_shapes, out_shapes, init_weight=None):

    # construct projector to align the feature dimension
    module = None
    if len(out_shapes) == 3:
        _, embed_tokens, embed_dim = out_shapes
        if len(in_shapes) == 4:
            module = CNN2ViTProjector(in_shapes, embed_tokens, embed_dim)
        elif len(in_shapes) == 3:
            module = ViT2ViTProjector(in_shapes, embed_tokens, embed_dim)
        else:
            raise ModuleNotFoundError

    elif len(out_shapes) == 4:
        _, _, h, w = out_shapes
        if len(in_shapes) == 3:
            module = ViT2CNNProjector(in_shapes, out_shapes)
        elif len(in_shapes) == 4:
            module = CNN2CNNProjector(in_shapes, out_shapes)
        else:
            raise ModuleNotFoundError

    # initialize the projector with specific function
    module.apply(init_weights)
    return module


class PLFEEncoder(nn.Module):
    def __init__(
        self,
        student_shapes: torch.Size,
        teacher_shapes: torch.Size,
        out_shapes: torch.Size,
        num_stages: int = 4,
    ) -> None:
        super(PLFEEncoder, self).__init__()

        # dimension, should be lower than input tensor's d1 from teacher/student transformer
        _, embed_tokens, embed_dim = out_shapes
        self.embed_tokens = embed_tokens
        self.embed_dim = embed_dim
        self.out_shapes = out_shapes

        # construct projector for each feature shape (b, c, h, w) -> (b, l, d)
        # for both teacher and student
        self.projector = nn.ModuleDict()
        for feat_shape in student_shapes:
            k = "x".join([str(i) for i in feat_shape[1:]])
            if k in self.projector:
                continue
            projector = get_projector_module(feat_shape, out_shapes)
            set_module_dict(self.projector, k, projector)

        for feat_shape in teacher_shapes:
            k = "x".join([str(i) for i in feat_shape[1:]])
            if k in self.projector:
                continue
            projector = get_projector_module(feat_shape, out_shapes)
            set_module_dict(self.projector, k, projector)

        print("=" * 100)
        print("PLFEEncoder.module_dict", self.projector.keys())
        for module_k in self.projector.keys():
            module = get_module_dict(self.projector, module_k)
            print(f"{module_k}, {module}, {get_module_parameters(module)}")

        #
        self.encoder = nn.Sequential(
            *[Block(dim=embed_dim, num_heads=4) for _ in range(num_stages)]  # todo: check this
        )
        for n, m in self.encoder.named_modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d):
                # trunc_normal_(m.weight, std=0.02)
                # if m.bias is not None:
                #     nn.init.zeros_(m.bias)
                torch.nn.init.trunc_normal_(m)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        print("encoder", get_module_parameters(self.encoder))

    def forward(self, x: torch.Tensor):
        # check the format of input tensor
        # if the shape of input tensor is from cnn (b, c, h, w), reshape it to (b, l, d)
        # if the tensor is from transformer (b, l, d1), downsample to (b, l, d)
        k = "x".join([str(i) for i in x.shape[1:]])
        projector = get_module_dict(self.projector, k)
        embed = projector(x)
        # print("PLFEEncoder", x.shape, embed.shape)

        # forward transformer encoder
        embed = self.encoder(embed)
        return embed


class PLFEDecoder(nn.Module):
    def __init__(
        self,
        is_cnn: bool,
        feat_shapes: torch.Size,
        embed_shape: torch.Size,
        num_stages: int = 4,
        *args,
        **kwargs,
    ) -> None:
        super(PLFEDecoder, self).__init__()
        self.embed_shape = embed_shape
        _, embed_tokens, embed_dim = self.embed_shape

        # construct projectors
        self.projector = nn.ModuleDict()
        for feat_shape in feat_shapes:
            k = "x".join([str(i) for i in feat_shape[1:]])
            if k in self.projector:
                continue

            out_shape = feat_shape
            in_shape = embed_shape
            if is_cnn and len(embed_shape) == 3:
                h = w = int(embed_tokens**0.5)
                in_shape = torch.Size([1, embed_dim, h, w])
            projector = get_projector_module(in_shape, out_shape)
            set_module_dict(self.projector, k, projector)

        # for feat_shape in teacher_shapes:
        #     k = "x".join([str(i) for i in feat_shape[1:]])
        #     if k in self.projector:
        #         continue
        #     projector = get_projector_module(in_shapes, feat_shape)
        #     set_module_dict(self.projector, k, projector)

        print("=" * 100)
        print("PLFEDecoder.module_dict", self.projector.keys())
        for module_k in self.projector.keys():
            module = get_module_dict(self.projector, module_k)
            print(f"{module_k}, {module}, {get_module_parameters(module)}")

        # construct transformer decoder
        decoder = None
        if is_cnn:
            c = embed_dim
            decoder = nn.Sequential(
                # using depthwise convolution to save vram usage
                # depthwise convolution
                nn.BatchNorm2d(c),
                nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, groups=c, bias=False),
                nn.Conv2d(c, c, kernel_size=1, stride=1, bias=False),
                # normal convolution
                # nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, bias=False),
                # nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU(inplace=True),
                # nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, bias=False),
                # nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, bias=False),
                # nn.BatchNorm2d(c, affine=True),
                # nn.ReLU(inplace=True),
            )
        else:
            decoder = nn.Sequential(*[Block(dim=embed_dim, num_heads=4) for _ in range(num_stages)])  # todo: check this
        self.is_cnn = is_cnn
        self.decoder = decoder
        self.decoder.apply(init_weights)
        print("decoder", get_module_parameters(self.decoder))

    def forward(self, embed: torch.Tensor, out_shape: torch.Size) -> torch.Tensor:
        # project the embed (b, l, d) to cnn (b, c, h, w), or vit & mlp (b, l, d1)

        # if decoder is a cnn model
        # reshape the embed to the shape of prior-introduced decoder
        if self.is_cnn:
            b, l, d = embed.shape
            w = h = int(math.sqrt(l))
            embed = torch.reshape(embed, (b, h, w, d))
            embed = embed.permute(0, 3, 1, 2)  # (b, d, h, w)

        # then forward piror-introduced decoder
        x = self.decoder(embed)

        # finally, reshape the decoder output to out_shape
        k = "x".join([str(i) for i in out_shape[1:]])
        projector = get_module_dict(self.projector, k)
        x = projector(x)

        return x


def get_positional_encoding(d_model: int, max_len: int = 5000):
    # Empty encodings vectors
    encodings = torch.zeros(max_len, d_model)
    # Position indexes
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    # $2 * i$
    two_i = torch.arange(0, d_model, 2, dtype=torch.float32)
    # $10000^{\frac{2i}{d_{model}}}$
    div_term = torch.exp(two_i * -(math.log(10000.0) / d_model))
    # $PE_{p,2i} = sin\Bigg(\frac{p}{10000^{\frac{2i}{d_{model}}}}\Bigg)$
    encodings[:, 0::2] = torch.sin(position * div_term)
    # $PE_{p,2i + 1} = cos\Bigg(\frac{p}{10000^{\frac{2i}{d_{model}}}}\Bigg)$
    encodings[:, 1::2] = torch.cos(position * div_term)

    # Add batch dimension
    encodings = encodings.unsqueeze(1).requires_grad_(False)

    return encodings


class AFD(nn.Module):
    def __init__(self, embed_shapes, num_stages, dim) -> None:
        super(AFD, self).__init__()
        self.qk_dim = dim
        self.embed_shapes = embed_shapes
        self.num_stages = num_stages

        # FFNs
        self.dim = dim
        self.to_q = nn.Sequential(
            nn.LayerNorm(self.embed_shapes[-1]),
            nn.Linear(self.embed_shapes[-1], dim),
        )
        self.to_k = nn.Sequential(
            nn.LayerNorm(self.embed_shapes[-1]),
            nn.Linear(self.embed_shapes[-1], dim),
        )

        # learnable position embeddings
        # which are utilized to share common information over different instances
        self.p_t = nn.Parameter(torch.Tensor(self.num_stages, self.qk_dim))
        self.p_s = nn.Parameter(torch.Tensor(self.num_stages, self.qk_dim))
        torch.nn.init.xavier_normal_(self.p_t)
        torch.nn.init.xavier_normal_(self.p_s)

    # def cal_diff(self, v_s, v_t, att):
    #     diff = (v_s - v_t.unsqueeze(1)).pow(2).mean(2)
    #     diff = torch.mul(diff, att).sum(1).mean()
    #     return diff

    def forward(self, feat_s: torch.Tensor, feat_t: torch.Tensor):
        # giving feat_s and feat_t, the student and teacher features, we calculate the attentnio matrix α
        # by utilizing αt, the teacher feature, enables to transfer its knowledge selectively to student features.
        # return loss

        assert feat_s.shape == feat_t.shape, "feat_s and feat_t should have the same shape"
        _, b, l, d = feat_s.shape  # [b, n(num_stages * embed_shape[0]), l, d]
        q = self.to_q(feat_t)  # [b, n, l, d]
        k = self.to_k(feat_s)  # [b, n, l, d]

        # calculate attention matrix
        pe = torch.matmul(self.p_t, self.p_s.t())
        # pe = get_positional_encoding(self.num_stages)

        logits = torch.einsum("bild,bjld->bij", q, k)  # [b, n, n]
        logits = torch.add(logits, pe) / np.sqrt(self.qk_dim)
        atts = F.softmax(logits, dim=2)  # b x n x n

        losses = []
        for i in range(atts.shape[1]):
            # fused_feat_s = torch.matmul(atts[:, i, :], feat_s)  # b x d
            # print(atts.shape, atts[:, i : i + 1, :].shape, feat_s.shape)
            fused_feat_s = torch.einsum("bci,bjld->bcld", atts[:, i : i + 1, :], feat_s)
            # print(fused_feat_s.shape, feat_t[:, i, :, :].shape)
            losses.append(F.mse_loss(fused_feat_s, feat_t[:, i : i + 1, :, :]))
        return sum(losses)
