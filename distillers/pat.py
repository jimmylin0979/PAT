import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
from timm.models.layers import PatchEmbed
from timm.models.layers import _assert, trunc_normal_
from timm.models.vision_transformer import Block as ViTBlock
from timm.models.swin_transformer import SwinTransformerBlock
from timm.models.mlp_mixer import MixerBlock
from timm.models.convnext import ConvNeXtBlock

from typing import List
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from ._base import BaseDistiller
from .registry import register_distiller
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

###############
### Lossess ###
###############


def hcl_loss(feat_s: torch.Tensor, feat_t: torch.Tensor) -> torch.Tensor:
    """
    Calculate the Hierarchical MSE Loss between two sets of features.
    The function rearranges the input features if they are in 3D format,
        and then computes the MSE loss at multiple spatial resolutions to incorporate hierarchical information.

    The final loss value is normalized by the total count of levels considered.

    Parameters:
    - feat_s (torch.Tensor): Tensor of shape (N, C, H, W) representing the source features.
    - feat_t (torch.Tensor): Tensor of shape (N, C, H, W) representing the target features.

    Returns:
    - torch.Tensor: Calculated loss value based on the MSE loss between the source and target features at different levels of spatial resolution.
    """

    if len(feat_s.shape) == 3:
        b, l, d = feat_s.shape
        feat_s = rearrange(feat_s, "b (h w) d -> b d h w", h=int(math.sqrt(l)))

    if len(feat_t.shape) == 3:
        b, l, d = feat_t.shape
        feat_t = rearrange(feat_t, "b (h w) d -> b d h w", h=int(math.sqrt(l)))

    n, c, h, w = feat_s.shape
    loss = F.mse_loss(feat_s, feat_t, reduction="mean")
    cnt = 1.0
    tot = 1.0
    for l in [4, 2, 1]:
        if l >= h:
            continue
        tmpfs = F.adaptive_avg_pool2d(feat_s, (l, l))
        tmpft = F.adaptive_avg_pool2d(feat_t, (l, l))
        cnt /= 2.0
        loss += F.mse_loss(tmpfs, tmpft, reduction="mean") * cnt
        tot += cnt
    loss = loss / tot
    return loss


###############
###   pat   ###
###############

NUM_STAGES = 4  # the max stage number


@register_distiller
class PAT(BaseDistiller):
    requires_feat = True

    def __init__(self, student, teacher, criterion, input_size, args, **kwargs):
        super(PAT, self).__init__(student, teacher, criterion, args)

        ### calculate student/teacher shapes ###
        self.student.to("cuda")
        self.teacher.to("cuda")
        feat_s, feat_t = None, None
        with torch.no_grad():
            c, h, w = input_size
            x = torch.rand(1, c, h, w).cuda()
            _, feat_s = self.student(x, requires_feat=True)
            _, feat_t = self.teacher(x, requires_feat=True)

        student_shapes = []
        teacher_shapes = []
        num_stages = len(self.args.pat_afp_stage)
        for stage in range(1, NUM_STAGES + 1):
            idx_t, _ = self.teacher.stage_info(stage)
            feat_t_shape = feat_t[idx_t].shape
            if len(feat_t_shape) == 3 and not math.sqrt(feat_t_shape[1]).is_integer():
                b, l, d = feat_t_shape
                feat_t_shape = torch.Size((b, l - 1, d))
            teacher_shapes.append(feat_t_shape)

            idx_s, _ = self.student.stage_info(stage)
            feat_s_shape = feat_s[idx_s].shape
            if len(feat_s_shape) == 3 and not math.sqrt(feat_s_shape[1]).is_integer():
                b, l, d = feat_s_shape
                feat_t_shape = torch.Size((b, l - 1, d))
            student_shapes.append(feat_s_shape)

        ### install peft blocks into teacher model before each stages ###
        # [  ] case 1: Visual Prompt Tuning
        # [vv] case 2: Pro-tuning: Unified Prompt Tuning for Vision Tasks
        # [  ] case 3: Side-tuning
        # [  ] case 4: CST
        self.prompt_blocks = None
        prompt_shapes = self.forward_teacher_with_prompts(x, None, None, True)

        prompt_blocks = []
        for stage in range(1, NUM_STAGES + 1):
            if stage not in self.args.pat_afp_stage:
                prompt_blocks.append(None)
                continue
            stage_i = stage - 1
            prompt_shape = prompt_shapes[stage_i]
            module = PromptBlock(in_shapes=teacher_shapes[stage_i], out_shapes=prompt_shape)
            prompt_blocks.append(module)
        self.prompt_blocks = nn.ModuleList(prompt_blocks)
        self.prev_feat_student = None
        self.prev_feat_teacher = None

        ### construct region-aware distillation module ###
        num_queries = self.args.pat_raa_num_queries  # num_queries = num_stages(4) * num_queries_per_stage
        dim = self.args.pat_raa_dim  # dim = 512
        self.attention_blending = RegionAttention(
            student_shapes, teacher_shapes, num_queries=num_queries, dim=dim, heads=8, dropout=0.0
        )

        ### construct teacher-prior projector ###
        self.aligners = nn.ModuleList()
        for stage in range(1, NUM_STAGES + 1):
            idx_t, _ = self.teacher.stage_info(stage)
            # idx_s, _ = self.student.stage_info(stage)

            # just a simple alignment block
            module = get_projector_module([1, int(num_queries / NUM_STAGES), dim], teacher_shapes[stage - 1])

            # # teacher-prior alignment block
            # if isinstance(
            #     self.teacher,
            #     (timm.models.swin_transformer.SwinTransformer, timm.models.vision_transformer.VisionTransformer),
            # ):
            #     module = SimpleSwinBlock([1, int(num_queries / num_stages), dim], teacher_shapes[stage_i])
            # elif isinstance(self.teacher, timm.models.convnext.ConvNeXt):
            #     module = SimpleConvBlock([1, int(num_queries / num_stages), dim], teacher_shapes[stage_i])
            # elif isinstance(self.teacher, timm.models.mlp_mixer.MlpMixer):
            #     module = SimpleMLPBlock([1, int(num_queries / num_stages), dim], teacher_shapes[stage_i])
            # else:
            #     raise NotImplementedError

            self.aligners.append(module)

    def forward_with_prompts(self, x, stage_idx, feat_student, feat_teacher):

        # early return if the prompt block is still not yet being initialized
        # should only occus when doing forward_teacher_with_prompts() with requires_shape = True
        if self.prompt_blocks is None:
            return x

        fgap = None
        # Enable feedback from student to forward prompt blocks
        if feat_teacher != None and feat_student != None:
            fgap = feat_student[stage_idx] - feat_teacher[stage_idx]
        x = self.prompt_blocks[stage_idx](x, fgap)
        return x

    def forward_teacher_with_prompts(self, x, feat_student, feat_teacher, requires_shape: bool = False):

        # Pro-tuning
        # make sure teacher model is not trainable, but the prompt blocks are
        self.teacher.eval()
        if self.prompt_blocks is not None:
            self.prompt_blocks.train()

        # collect the stage idx of teacher model
        stage_idxs = []
        stage_idx = -1
        for stage in range(1, NUM_STAGES + 1):
            idx, _ = self.teacher.stage_info(stage)
            stage_idxs.append(idx)

        # collect the input shape of prompt
        prompt_shapes = []

        #
        # feat_idx is the index of the teacher feature layer that are going to be executed (but currently still not !)
        feat, feat_idx = [], 0
        if isinstance(self.teacher, timm.models.vision_transformer.VisionTransformer):

            x = self.teacher.patch_embed(x)
            x = torch.cat((self.teacher.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = self.teacher.pos_drop(x + self.teacher.pos_embed)
            for blk in self.teacher.blocks:
                # prompt interaction
                # we should perform prompt tuning before each stage, i.e., before the time that next stage is going to start
                if stage_idx == -1 or feat_idx == stage_idxs[stage_idx] + 1:
                    prompt_shapes.append(x.shape)
                    stage_idx += 1
                    if stage_idx + 1 in self.args.pat_afp_stage:
                        x = self.forward_with_prompts(x, stage_idx, feat_student, feat_teacher)

                x = blk(x)
                feat.append(x)
                feat_idx += 1

            x = self.teacher.norm(x)

            # forward head
            x = self.teacher.forward_head(x, pre_logits=True)
            feat.append(x)
            x = self.teacher.head(x)

        elif isinstance(self.teacher, timm.models.swin_transformer.SwinTransformer):

            x = self.teacher.patch_embed(x)
            if self.teacher.absolute_pos_embed is not None:
                x = x + self.teacher.absolute_pos_embed
            x = self.teacher.pos_drop(x)
            for layers in self.teacher.layers:
                for layer in layers.blocks:
                    # prompt interaction
                    if stage_idx == -1 or feat_idx == stage_idxs[stage_idx] + 1:
                        prompt_shapes.append(x.shape)
                        stage_idx += 1
                        if stage_idx + 1 in self.args.pat_afp_stage:
                            x = self.forward_with_prompts(x, stage_idx, feat_student, feat_teacher)

                    x = layer(x)
                    feat.append(x)
                    feat_idx += 1
                if layers.downsample is not None:
                    x = layers.downsample(x)

            # we do not insert the prompt block at the last stage,
            #   as we aim to make prompt focus on making intermidiate feature more student-friendly
            # x = self.prompt_blocks[-1](x)
            x = self.teacher.norm(x)  # B L C
            feat.append(x)  # idx=12/24, for debug

            # forward head
            x = self.teacher.forward_head(x, pre_logits=True)
            feat.append(x)
            x = self.teacher.head(x)

        elif isinstance(self.teacher, timm.models.convnext.ConvNeXt):

            x = self.teacher.stem(x)
            for stage in self.teacher.stages:
                # prompt interaction
                if stage_idx == -1 or feat_idx == stage_idxs[stage_idx] + 1:
                    prompt_shapes.append(x.shape)
                    stage_idx += 1
                    if stage_idx + 1 in self.args.pat_afp_stage:
                        x = self.forward_with_prompts(x, stage_idx, feat_student, feat_teacher)

                x = stage(x)
                feat.append(x)
                feat_idx += 1

            x = self.teacher.norm_pre(x)

            # forward head
            x = self.teacher.forward_head(x, pre_logits=True)
            feat.append(x)
            x = self.teacher.head.fc(x)

        elif isinstance(self.teacher, timm.models.mlp_mixer.MlpMixer):

            x = self.teacher.stem(x)
            for blk in self.teacher.blocks:
                # prompt interaction
                if stage_idx == -1 or feat_idx == stage_idxs[stage_idx] + 1:
                    prompt_shapes.append(x.shape)
                    stage_idx += 1
                    if stage_idx + 1 in self.args.pat_afp_stage:
                        x = self.forward_with_prompts(x, stage_idx, feat_student, feat_teacher)

                x = blk(x)
                feat.append(x)
                feat_idx += 1

            x = self.teacher.norm(x)

            # forward head
            if self.teacher.global_pool == "avg":
                x = x.mean(dim=1)
            feat.append(x)
            x = self.teacher.head(x)

        elif isinstance(self.teacher, timm.models.resnet.ResNet):

            x = self.teacher.conv1(x)
            x = self.teacher.bn1(x)
            x = self.teacher.act1(x)
            x = self.teacher.maxpool(x)

            # stage 1
            # prompt interaction
            if stage_idx == -1 or feat_idx == stage_idxs[stage_idx] + 1:
                prompt_shapes.append(x.shape)
                stage_idx += 1
                if stage_idx + 1 in self.args.pat_afp_stage:
                    x = self.forward_with_prompts(x, stage_idx, feat_student, feat_teacher)
            x = self.teacher.layer1(x)
            feat.append(x)
            feat_idx += 1

            # stage 2
            # prompt interaction
            if stage_idx == -1 or feat_idx == stage_idxs[stage_idx] + 1:
                prompt_shapes.append(x.shape)
                stage_idx += 1
                if stage_idx + 1 in self.args.pat_afp_stage:
                    x = self.forward_with_prompts(x, stage_idx, feat_student, feat_teacher)
            x = self.teacher.layer2(x)
            feat.append(x)
            feat_idx += 1

            # stage 3
            # prompt interaction
            if stage_idx == -1 or feat_idx == stage_idxs[stage_idx] + 1:
                prompt_shapes.append(x.shape)
                stage_idx += 1
                if stage_idx + 1 in self.args.pat_afp_stage:
                    x = self.forward_with_prompts(x, stage_idx, feat_student, feat_teacher)
            x = self.teacher.layer3(x)
            feat.append(x)
            feat_idx += 1

            # stage 4
            # prompt interaction
            if stage_idx == -1 or feat_idx == stage_idxs[stage_idx] + 1:
                prompt_shapes.append(x.shape)
                stage_idx += 1
                if stage_idx + 1 in self.args.pat_afp_stage:
                    x = self.forward_with_prompts(x, stage_idx, feat_student, feat_teacher)
            x = self.teacher.layer4(x)
            feat.append(x)
            feat_idx += 1

            # forward head
            x = self.teacher.forward_head(x, pre_logits=True)
            feat.append(x)
            x = self.teacher.fc(x)

        else:
            raise NotImplementedError

        if requires_shape:
            return prompt_shapes
        return x, feat

    def forward(self, image: torch.Tensor, label, epoch: int, *args, **kwargs):

        ### student forward ###
        logits_student, feat_student = self.student(image, requires_feat=True)

        ### teacher forward ###
        with torch.no_grad():
            logits_teacher_woPrompt, feat_teacher_woPrompt = self.teacher(image, requires_feat=True)

        # forward another teacher feature with prompt incorporate
        self.teacher.eval()
        logits_teacher, feat_teacher = self.forward_teacher_with_prompts(
            image, self.prev_feat_student, self.prev_feat_teacher
        )

        ### pat module forward ###
        # extract the stage feature from teacher & student
        feat_s, feat_t = [], []
        for stage in range(1, NUM_STAGES + 1):
            idx_t, _ = self.teacher.stage_info(stage)
            idx_s, _ = self.student.stage_info(stage)
            feat_t.append(feat_teacher[idx_t])
            feat_s.append(feat_student[idx_s])

        # blending the stage feature to bring T/S feature closer
        # return List[torch.Tensor] num_stages * tensor(b, num_queries_per_stage, dim)
        feat_s_blended = self.attention_blending(feat_s, feat_t)

        loss_pat_mse = 0.0
        prev_feat_student, prev_feat_teacher = [], []
        for stage in range(len(feat_t)):
            feat_s_aligned = self.aligners[stage](feat_s_blended[stage].squeeze(0))
            feat_t_i = feat_t[stage]
            if len(feat_t_i.shape) == 3 and not math.sqrt(feat_t_i.shape[1]).is_integer():
                feat_t_i = feat_t_i[:, :-1, :]

            # loss_pat_mse = F.mse_loss(feat_s_aligned, feat_t_i)
            loss_pat_mse += hcl_loss(feat_s_aligned, feat_t_i)
            prev_feat_student.append(feat_s_aligned.detach())
            prev_feat_teacher.append(feat_t_i.detach())

        # update previous feature, which is the feedback from student for prompt block on next epoch
        self.prev_feat_student = prev_feat_student
        self.prev_feat_teacher = prev_feat_teacher

        ### losses_dict construction ###
        loss_pat_mse = self.args.pat_mse_loss_weight * loss_pat_mse
        loss_pat_reg = self.args.pat_reg_loss_weight * kd_loss(logits_teacher, logits_teacher_woPrompt)
        loss_gt = self.args.gt_loss_weight * self.criterion(logits_student, label)
        loss_kd = self.args.kd_loss_weight * kd_loss(logits_student, logits_teacher_woPrompt, self.args.kd_temperature)
        losses_dict = {
            "loss_gt": loss_gt,
            "loss_kd": loss_kd,
            "loss_pat_mse": loss_pat_mse,
            "loss_pat_reg": loss_pat_reg,
        }
        return logits_student, losses_dict


#########################
### Pro-tuning Blocks ###
#########################


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block.

    Args:
        in_channels (int): Number of input channels.
        ratio (int): Reduction ratio for the squeeze operation.

    Attributes:
        squeeze (nn.AdaptiveAvgPool2d): Squeeze operation to reduce spatial dimensions.
        compress (nn.Conv2d): Convolutional layer to reduce the number of channels.
        excitation (nn.Conv2d): Convolutional layer to increase the number of channels back to the original.

    Methods:
        forward(x): Forward pass of the SEBlock, applies squeeze, excitation, and activation functions.

    Returns:
        torch.Tensor: Output tensor after applying the Squeeze-and-Excitation operations.

    References:
        https://zhuanlan.zhihu.com/p/263537813
    """

    def __init__(self, in_channels: int, ratio: int):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress = nn.Conv2d(in_channels, in_channels // ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_channels // ratio, in_channels, 1, 1, 0)

    def forward(self, x: torch.Tensor):
        out = self.squeeze(x)
        out = self.compress(out)
        out = F.relu(out)
        out = self.excitation(out)
        return F.sigmoid(out)


class PromptBlock(nn.Module):
    def __init__(self, in_shapes: torch.Size, out_shapes: torch.Size) -> None:
        """
        Initialize the PromptBlock module.

        Args:
            in_shapes (torch.Size): Input shape of the module.
            out_shapes (torch.Size): Output shape of the module.

        Attributes:
            fb_aligner (nn.Identity): Identity module for alignment.
            fusion_block (nn.Sequential): Sequential block for fusion operations.
            beta (nn.Parameter): Learnable parameter for blending.
            prompt_block (nn.Sequential): Sequential block for prompt operations.

        """
        super(PromptBlock, self).__init__()

        #
        if len(in_shapes) == 3:
            _, l, ic = in_shapes
            iw = ih = int(math.sqrt(l))
        else:
            _, ic, ih, iw = in_shapes

        if len(out_shapes) == 3:
            _, l, oc = out_shapes
            ow = oh = int(math.sqrt(l))
        else:
            _, oc, oh, ow = out_shapes

        ### Fusion Blocks ###
        self.fb_aligner = nn.Identity()
        if oh != ih or ow != iw or ic != oc:
            # construct the align block if any dimension is mismatch (-1, ic, ih, iw) -> (-1, oc, oh, ow)
            self.fb_aligner = get_projector_module(in_shapes, out_shapes)
        self.fusion_block = nn.Sequential(
            nn.Conv2d(2 * oc, oc, 1, 1, 0),
            nn.Conv2d(oc, oc, 5, 1, 2),
            nn.Conv2d(oc, oc, 1, 1, 0),
            SEBlock(oc, ratio=16),
            nn.Conv2d(oc, oc, 1, 1, 0),
        )

        ### Prompt Blocks ###
        # prompt block that follow the settings in Pro-tuning
        # Reference: https://paperswithcode.com/paper/pro-tuning-unified-prompt-tuning-for-vision
        self.beta = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        self.prompt_block = nn.Sequential(
            nn.Conv2d(oc, oc, 1, 1, 0, groups=4),
            nn.Conv2d(oc, oc, 5, 1, 2, groups=oc),
            nn.Conv2d(oc, oc, 1, 1, 0),
            SEBlock(oc, ratio=16),
            nn.Conv2d(oc, oc, 1, 1, 0, groups=4),
        )

        # FIXME: initialize with zero convolution (cause NAN during training however)
        # def init_weights(m):
        #     if isinstance(m, nn.Conv2d):
        #         m.bias.data.zero_()
        #         m.weight.data.fill_(1.0)

        # self.prompt_block.apply(init_weights)

    def forward(self, x: torch.Tensor, feedback: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor to the PromptBlock module.
            feedback (torch.Tensor, optional): Feedback tensor to align with the shape of the input tensor x. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after processing through the fusion block, prompt block, and blending with the input tensor x.

        Notes:
            - Aligns the feedback tensor to the shape of the input tensor x if feedback is provided.
            - Reshapes the input tensor to a 2D structure if the source model is transformer/mlp-based.
            - Applies fusion block to fuse the original teacher stage input and feedback feature if feedback is not None.
            - Applies the prompt block to the fused feature and blends it with the input tensor x.
            - Reshapes the output tensor back to a 1D sequence for sequenced-based models.
            - Concatenates the cls_token with the output tensor if cls_token is not None.
        """

        # first, align the feedback to the shape of teacher main branch feature x
        if feedback is not None:
            feedback = self.fb_aligner(feedback)

        # reshape the input tensor to 2D structure if the source model is transformer/mlp-based
        is_input_2d_stucture = True
        cls_token = None
        if len(x.shape) == 3:
            is_input_2d_stucture = False
            b, l, d = x.shape
            h = w = int(math.sqrt(l))
            if not math.sqrt(l).is_integer():  # remove cls token
                cls_token = x[:, -1, :]
                x = x[:, :-1, :]
            x = torch.permute(x, (0, 2, 1))  # (b, d, l) -> (b, c, l)
            x = torch.reshape(x, (b, d, h, w))  # (b, c, l) -> (b, c, h, w)

            if feedback is not None:
                feedback = torch.permute(feedback, (0, 2, 1))  # (b, d, l) -> (b, c, l)
                feedback = torch.reshape(feedback, (b, d, h, w))  # (b, c, l) -> (b, c, h, w)

        x_fusion = x
        if feedback is not None:
            # forward fusion block to fuse the original teacher stage input & feebback feature
            x_fusion = self.fusion_block(torch.concat((x, feedback), dim=1))  # (b, 2c, h, w) -> (b, c, h, w)

        # forward through prompt block and then blend the feature
        x_prompt = self.prompt_block(x_fusion)
        x = x + self.beta * x_prompt

        # reshape back to 1D sequence for sequenced-based model
        if not is_input_2d_stucture:
            b, d, h, w = x.shape
            x = torch.reshape(x, (b, h * w, d))

        if cls_token is not None:
            x = torch.concat((x, cls_token.unsqueeze(1)), dim=1)

        return x


################################
### Heterogeneous Projectors ###
################################


class CNN2ViTProjector(nn.Module):
    def __init__(self, in_shapes, embed_tokens: int, embed_dim: int) -> None:
        super(CNN2ViTProjector, self).__init__()
        _, c, h, w = in_shapes
        assert h == w, "feature width should be as same as the height"

        self.h = int(embed_tokens**0.5)  # the number of patch per side if we reshape the sequence into square
        patch_size = int(math.ceil(h / (embed_tokens**0.5)))  # the number of pixels per side of a patch

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

        # TODO: Progressively upsampling
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


############################
### Teacher-piror Blocks ###
############################

# FUTURE: A simple conv block is good enough, and teacher-prior block costs too many VRAM but improve a little

# class SimpleConvBlock(nn.Module):
#     def __init__(self, in_shape, out_shape, *args, **kwargs) -> None:
#         super(SimpleConvBlock, self).__init__(*args, **kwargs)

#         self.projector = get_projector_module(in_shape, out_shape)

#         b, c, h, w = out_shape
#         self.padding = nn.ZeroPad2d(3)
#         self.module = ConvNeXtBlock(dim=c, drop_path=0.1)  # kernerl_size 7, stride 1

#     def forward(self, x):
#         x = self.projector(x)
#         x = self.module(x)
#         return x


# class SimpleSwinBlock(nn.Module):
#     def __init__(self, in_shape, out_shape, *args, **kwargs) -> None:
#         super(SimpleSwinBlock, self).__init__(*args, **kwargs)

#         self.projector = get_projector_module(in_shape, out_shape)

#         b, l, d = out_shape
#         h = w = int(math.sqrt(l))
#         self.module = SwinTransformerBlock(
#             dim=d,
#             input_resolution=(h, w),
#             num_heads=4,
#             head_dim=512,
#             window_size=7,
#             drop=0.1,
#             attn_drop=0.1,
#             drop_path=0.1,
#         )

#     def forward(self, x):
#         x = self.projector(x)
#         x = self.module(x)
#         return x


# class SimpleMLPBlock(nn.Module):
#     def __init__(self, in_shape, out_shape, *args, **kwargs) -> None:
#         super(SimpleMLPBlock, self).__init__(*args, **kwargs)

#         self.projector = get_projector_module(in_shape, out_shape)

#         b, l, d = out_shape
#         self.module = MixerBlock(dim=d, seq_len=l, drop=0.1, drop_path=0.1)

#     def forward(self, x):
#         x = self.projector(x)
#         x = self.module(x)
#         return x


##############################
### Region-Aware Attention ###
##############################


class RAProjector(nn.Module):
    def __init__(self, feat_shape: torch.Size, patch_length: int, embed_dim: int, is_q: bool) -> None:
        """
        Projector class for Region-Aware Attention mechanism.
            , aiming to project the input feature tensor into a patch-based feature tensor (1D) (b, patch_length,embed_dim).

        Args:
            feat_shape (torch.Size): The shape of the input feature tensor.
            patch_length (int): The length of the patch of output feature.
            embed_dim (int): The dimension of the embedding.
            # is_q (bool): A flag indicating whether the projector is for query (Q) or key-value (KV).

        Attributes:
            patch_length (int): The length of the patch.
            patch_h (int): The number of patches per size.
            embed_dim (int): The dimension of the embedding.
            img_size (int): The size of the image.
            patch_size (int): The size of the patch.
            projector (PatchEmbed): The PatchEmbed module for projecting the input tensor.

        Methods:
            forward(x): Forward pass through the projector.

        Raises:
            NotImplementedError: If the input feature shape is not supported.

        Note:
            The projector reshapes the input tensor into a 2D grid shape and projects it using PatchEmbed.

        Reference:
            https://github.com/facebookresearch/dino/issues/33
        """
        super(RAProjector, self).__init__()

        # set output embedding dimension
        self.patch_length = patch_length
        self.patch_h = int(math.sqrt(patch_length))  # the number of patches per size
        self.embed_dim = embed_dim
        # self.is_q = is_q

        ### Projector ###
        # first, we reshape the input tensor into 2D grid shape, and shrink the feature map by PatchEmbed to a pre-defined size
        self.img_size = None
        self.patch_size = None
        c = 0
        if len(feat_shape) == 4:
            b, c, h, w = feat_shape
            self.patch_size = int(math.ceil(h / self.patch_h))  # the number of pixels per patch
            self.img_size = int(self.patch_size * self.patch_h)
        elif len(feat_shape) == 3:
            b, l, c = feat_shape
            h = int(math.sqrt(l))
            self.patch_size = int(math.ceil(h / self.patch_h))
            self.img_size = int(self.patch_size * self.patch_h)
        else:
            raise NotImplementedError

        # Reference Why do not use norm in PatchEmbed, https://github.com/facebookresearch/dino/issues/33
        assert self.img_size is not None, "img_size should not be None"
        projector = PatchEmbed(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=c,
            embed_dim=embed_dim,
        )
        self.projector = projector

        # ### QKV Linear ###
        # # second, we project the input tensor to Q/KV, and then normalize the Q/KV as input to attention module
        # self.norm = nn.LayerNorm(embed_dim)
        # self.to_q, self.to_kv = None, None
        # if self.is_q:
        #     self.to_q = nn.Linear(embed_dim, embed_dim, bias=False)
        # else:
        #     self.to_kv = nn.Linear(embed_dim, embed_dim * 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # reshape the input tensor to 2D structure if the source model is transformer/mlp-based
        if len(x.shape) == 3:  # (b, l, d)
            b, l, d = x.shape
            h = w = int(math.sqrt(l))
            if not math.sqrt(l).is_integer():  # remove cls token
                x = x[:, :-1, :]
            x = torch.permute(x, (0, 2, 1))  # (b, d, l) -> (b, c, l)
            x = torch.reshape(x, (b, d, h, w))  # (b, c, h, w)

        # reshape the input tensor to the shape that fit the patch embed module
        if x.shape[2] < self.patch_h:
            x = torch.nn.functional.interpolate(x, size=self.patch_h, mode="nearest")
        if x.shape[2] < self.img_size:
            x = torch.nn.functional.interpolate(x, size=self.patch_size * self.patch_h, mode="nearest")

        # forwrad through patch embed module
        embed = self.projector(x)
        # embed = self.norm(embed)
        # if self.is_q:
        #     embed = self.to_q(embed)
        # else:
        #     embed = self.to_kv(embed)
        return embed


# Fixed position embedding
# Reference: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/positional_encoding.py
def get_positional_encoding(d_model: int, max_len: int = 5000):
    """
    Generate positional encodings for transformer models.

    Args:
        d_model (int): The number of expected features in the input (required).
        max_len (int): The maximum length of the sequence (default is 5000).

    Returns:
        torch.Tensor: A tensor of shape (max_len, 1, d_model) containing the positional encodings.

    Example:
        >>> get_positional_encoding(512, 100)
    """

    encodings = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    two_i = torch.arange(0, d_model, 2, dtype=torch.float32)
    div_term = torch.exp(two_i * -(math.log(10000.0) / d_model))
    encodings[:, 0::2] = torch.sin(position * div_term)
    encodings[:, 1::2] = torch.cos(position * div_term)

    # Add batch dimension
    encodings = encodings.unsqueeze(1).requires_grad_(False)
    return encodings


class RegionAttention(nn.Module):
    """
    A receptive field aware attention module,
        aiming to solve the view mismatch problem between heterogeneous architecture.
    """

    def __init__(
        self,
        feat_student_shapes: List[torch.Size],
        feat_teacher_shapes: List[torch.Size],
        num_queries: int,
        dim: int,
        heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        """
        Initialize the Region-Aware Attention module.

        Args:
            feat_student_shapes (List[torch.Size]): List of shapes of the student features.
            feat_teacher_shapes (List[torch.Size]): List of shapes of the teacher features.
            num_queries (int): Number of query tokens incorporated in attention.
            dim (int): Embedding dimension of each token.
            heads (int, optional): Number of heads for Multi-Head Self-Attention. Defaults to 8.
            dropout (float, optional): Dropout probability. Defaults to 0.0.
        """
        super(RegionAttention, self).__init__()

        self.num_stages = len(feat_student_shapes)
        self.num_queries = num_queries
        self.num_queries_per_stage = self.num_queries // self.num_stages
        self.heads = heads
        self.dim = dim
        self.scale = (self.dim // self.heads) ** -0.5

        ### Projectors ###
        # construct projector to project teacher/student feature to a predefine sequence shape
        self.projector_s = nn.ModuleList()
        for i, feat_shape in enumerate(feat_student_shapes):
            if len(feat_shape) == 3:  # (b, l, d)
                b, l, d = feat_shape
                if not math.sqrt(feat_shape[1]).is_integer():
                    l -= 1
                feat_shape = torch.Size((b, l, d))
            module = RAProjector(feat_shape, patch_length=self.num_queries_per_stage, embed_dim=dim, is_q=False)
            self.projector_s.append(module)

        # self.projector_t = nn.ModuleList()
        # for i, feat_shape in enumerate(feat_teacher_shapes):
        #     if len(feat_shape) == 3:  # (b, l, d)
        #         b, l, d = feat_shape
        #         if not math.sqrt(feat_shape[1]).is_integer():
        #             l -= 1
        #         feat_shape = torch.Size((b, l, d))
        #     module = RAProjector(feat_shape, patch_length=self.num_queries_per_stage, embed_dim=dim, is_q=True)
        #     self.projector_t.append(module)

        ### Attention ###
        # position embedding buffer
        self.register_buffer("positional_encodings", get_positional_encoding(self.dim))

        # attention submodule, such as normlization layer, q, k, v
        self.norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim)
        self.to_kv = nn.Linear(dim, dim * 2)
        self.attn_dropout = nn.Dropout(dropout)

        # TODO: is output projector needed ?
        # is_project_out = False
        # self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout)) if is_project_out else nn.Identity()

    def forward(self, feat_s: List[torch.Tensor], feat_t: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Perform the forward pass of the RegionAttention module.

        Args:
            feat_s (List[torch.Tensor]): List of student features.
            feat_t (List[torch.Tensor]): List of teacher features.

        Returns:
            List[torch.Tensor]: List of blended student tensors.
        """
        # extract query, key from teacher, student
        qs, ks, vs = [], [], []
        for i, (feat_s, feat_t) in enumerate(zip(feat_s, feat_t)):
            _q = _kv = self.projector_s[i](feat_s)
            _q = self.to_q(self.norm(_q))
            _k, _v = self.to_kv(self.norm(_kv)).chunk(2, dim=-1)
            qs.append(_q)
            ks.append(_k)
            vs.append(_v)

        qs = torch.stack(qs, dim=1).reshape(-1, self.num_queries, self.dim)
        ks = torch.stack(ks, dim=1).reshape(-1, self.num_queries, self.dim)
        vs = torch.stack(vs, dim=1).reshape(-1, self.num_queries, self.dim)

        # calculate attention matrix
        pe = self.positional_encodings[: qs.shape[0]].requires_grad_(False)
        qs = qs + pe
        ks = ks + pe
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), [qs, ks, vs])

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        output = torch.matmul(attn, v)
        output = rearrange(output, "b h n d -> b n (h d)")
        output = rearrange(output, "b (n q) d -> n b q d", n=self.num_stages)
        output = output.chunk(self.num_stages, dim=0)

        return output
