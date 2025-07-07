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
            # iw = ih = int(math.sqrt(l))
            factors = self.find_factors(int(l))
            iw = int(factors[len(factors) // 2])
            ih = int(l // iw)

        else:
            _, ic, ih, iw = in_shapes

        if len(out_shapes) == 3:
            _, l, oc = out_shapes
            # ow = oh = int(math.sqrt(l))
            factors = self.find_factors(int(l))
            ow = int(factors[len(factors) // 2])
            oh = int(l // iw)

        else:
            _, oc, oh, ow = out_shapes

        # self.iw, self.ih = iw, ih
        # self.ow, self.oh = ow, oh

        ### Fusion Blocks ###
        self.fb_aligner = nn.Identity()
        if oh != ih or ow != iw or ic != oc:
            # construct the align block if any dimension is mismatch (-1, ic, ih, iw) -> (-1, oc, oh, ow)
            # FIXME: in_shapes should be i-th stage feedback feature
            # self.fb_aligner = get_projector_module(in_shapes, out_shapes)
            self.fb_aligner = nn.Sequential(nn.Conv2d(ic, oc, 1, 1, 0, bias=False))
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

    def find_factors(self, num):
        #
        i, factors = 2, [1, num]
        while i < num:
            if num % i == 0:
                factors.append(i)
                factors.append(num // i)
            if i * i >= num:
                break
            i += 1

        factors.sort()
        return factors

    def forward(self, x: torch.Tensor, h, w, feedback: torch.Tensor = None) -> torch.Tensor:
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
            feedback = F.interpolate(feedback, size=(h, w))
            feedback = self.fb_aligner(feedback)

        # reshape the input tensor to 2D structure if the source model is transformer/mlp-based
        is_input_2d_stucture = True
        cls_token = None
        if len(x.shape) == 3:
            is_input_2d_stucture = False
            b, l, d = x.shape
            # No cls token for transformer in downstream task
            # h = w = int(math.sqrt(l))
            # if not math.sqrt(l).is_integer():  # remove cls token
            #     cls_token = x[:, -1, :]
            #     x = x[:, :-1, :]
            x = torch.permute(x, (0, 2, 1))  # (b, d, l) -> (b, c, l)
            x = torch.reshape(x, (b, d, h, w))  # (b, c, l) -> (b, c, h, w)

            if feedback is not None and len(feedback.shape) == 3:
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
            b, d, _h, _w = x.shape
            x = torch.reshape(x, (b, _h * _w, d))

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


def init_weights(module):
    for n, m in module.named_modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
