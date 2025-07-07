import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from timm.models.layers import _assert, trunc_normal_


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, out_dim=None, norm_layer=nn.LayerNorm, act_layer=nn.Identity):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = out_dim or 2 * dim
        self.norm = norm_layer(4 * dim)

        in_features = 4 * dim
        self.reduction = nn.Linear(in_features, self.out_dim, bias=False)
        self.act = act_layer()

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        _assert(L == H * W, "input feature has wrong size")
        _assert(H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even.")

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)
        x = self.act(x)

        return x


class GAP1d(nn.Module):
    def __init__(self):
        super(GAP1d, self).__init__()

    def forward(self, x):
        return x.mean(1)


class TokenFilter(nn.Module):
    """remove cls tokens in forward"""

    def __init__(self, number=1, inverse=False, remove_mode=True):
        super(TokenFilter, self).__init__()
        self.number = number
        self.inverse = inverse
        self.remove_mode = remove_mode

    def forward(self, x):
        if self.inverse and self.remove_mode:
            x = x[:, : -self.number, :]
        elif self.inverse and not self.remove_mode:
            x = x[:, -self.number :, :]
        elif not self.inverse and self.remove_mode:
            x = x[:, self.number :, :]
        else:
            x = x[:, : self.number, :]
        return x


class TokenFnContext(nn.Module):
    def __init__(
        self,
        token_num=0,
        fn: nn.Module = nn.Identity(),
        token_fn: nn.Module = nn.Identity(),
        inverse=False,
    ):
        super(TokenFnContext, self).__init__()
        self.token_num = token_num
        self.fn = fn
        self.token_fn = token_fn
        self.inverse = inverse
        self.token_filter = TokenFilter(number=token_num, inverse=inverse, remove_mode=False)
        self.feature_filter = TokenFilter(number=token_num, inverse=inverse)

    def forward(self, x):
        tokens = self.token_filter(x)
        features = self.feature_filter(x)
        features = self.fn(features)
        if self.token_num == 0:
            return features

        tokens = self.token_fn(tokens)
        if self.inverse:
            x = torch.cat([features, tokens], dim=1)
        else:
            x = torch.cat([tokens, features], dim=1)
        return x


class LambdaModule(nn.Module):
    def __init__(self, lambda_fn):
        super(LambdaModule, self).__init__()
        self.fn = lambda_fn

    def forward(self, x):
        return self.fn(x)


class MyPatchMerging(nn.Module):
    def __init__(self, out_patch_num):
        super().__init__()
        self.out_patch_num = out_patch_num

    def forward(self, x):
        B, L, D = x.shape
        patch_size = int(L**0.5)
        assert patch_size**2 == L
        out_patch_size = int(self.out_patch_num**0.5)
        assert out_patch_size**2 == self.out_patch_num
        grid_size = patch_size // out_patch_size
        assert grid_size * out_patch_size == patch_size
        x = x.view(B, out_patch_size, grid_size, out_patch_size, grid_size, D)
        x = torch.einsum("bhpwqd->bhwpqd", x)
        x = x.reshape(shape=(B, out_patch_size**2, -1))
        return x


class SepConv(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=3, stride=2, padding=1, affine=True):
        #   depthwise and pointwise convolution, downsample by 2
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(
                channel_in,
                channel_in,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=channel_in,
                bias=False,
            ),
            nn.Conv2d(channel_in, channel_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                channel_in,
                channel_in,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                groups=channel_in,
                bias=False,
            ),
            nn.Conv2d(channel_in, channel_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_out, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)


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


def kd_loss(logits_student, logits_teacher, temperature=1.0):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="batchmean")
    loss_kd *= temperature**2
    return loss_kd


def is_cnn_model(distiller):
    if hasattr(distiller, "module"):
        _, sizes = distiller.module.stage_info(1)
    else:
        _, sizes = distiller.stage_info(1)
    if len(sizes) == 3:  # C H W
        return True
    elif len(sizes) == 2:  # L D
        return False
    else:
        raise RuntimeError("unknown model feature shape")


def set_module_dict(module_dict, k, v):
    if not isinstance(k, str):
        k = str(k)
    module_dict[k] = v


def get_module_dict(module_dict, k):
    if not isinstance(k, str):
        k = str(k)
    return module_dict[k]


def patchify(imgs, p):
    """
    imgs: (N, C, H, W)
    x: (N, L, patch_size**2, C)
    """
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    in_chans = imgs.shape[1]
    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], in_chans, h, p, w, p))
    x = torch.einsum("nchpwq->nhwpqc", x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * in_chans))
    return x


class Unpatchify(nn.Module):
    def __init__(self, p):
        super(Unpatchify, self).__init__()
        self.p = p

    def _unpatchify(self, x, p):
        """
        x: (N, L, patch_size**2 *C)
        imgs: (N, C, H, W)
        """
        # p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1], f"x.shape[1] = {x.shape[1]}, which is not match with h * w, {h} * {w}"

        x = x.reshape(shape=(x.shape[0], h, w, p, p, -1))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], -1, h * p, h * p))
        return imgs

    def forward(self, x):
        return self._unpatchify(x, self.p)


def get_module_parameters(module):
    num_p = 0
    for p in module.parameters():
        num_p += p.numel()
    return num_p


# Reference from https://github.com/jayroxis/CKA-similarity
class CKA(object):
    def __init__(self):
        pass

    def centering(self, K):
        n = K.shape[0]
        unit = np.ones([n, n])
        I = np.eye(n)
        H = I - unit / n
        return np.dot(np.dot(H, K), H)

    def rbf(self, X, sigma=None):
        GX = np.dot(X, X.T)
        KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
        if sigma is None:
            mdist = np.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= -0.5 / (sigma * sigma)
        KX = np.exp(KX)
        return KX

    def kernel_HSIC(self, X, Y, sigma):
        return np.sum(self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma)))

    def linear_HSIC(self, X, Y):
        L_X = X @ X.T
        L_Y = Y @ Y.T
        return np.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = np.sqrt(self.linear_HSIC(X, X))
        var2 = np.sqrt(self.linear_HSIC(Y, Y))
        return hsic / (var1 * var2)

    def kernel_CKA(self, X, Y, sigma=None):
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = np.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = np.sqrt(self.kernel_HSIC(Y, Y, sigma))
        return hsic / (var1 * var2)


class CudaCKA(object):
    def __init__(self, device):
        self.device = device

    def centering(self, K):
        # K [n, b, b]
        n, b, _ = K.shape
        unit = torch.ones([b, b], device=self.device)
        I = torch.eye(b, device=self.device)
        H = I - unit / b
        H = torch.unsqueeze(H, 0)
        H = H.repeat(n, 1, 1)
        return torch.matmul(torch.matmul(H, K), H)

    def rbf(self, X, sigma=None):
        GX = torch.matmul(X, X.T)
        KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
        if sigma is None:
            mdist = torch.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= -0.5 / (sigma * sigma)
        KX = torch.exp(KX)
        return KX

    def kernel_HSIC(self, X, Y, sigma):
        return torch.sum(self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma)))

    def linear_HSIC(self, X, Y):
        # X [n, b, d]
        # L_X = torch.matmul(X, X.T)
        # L_Y = torch.matmul(Y, Y.T)

        L_X = torch.einsum("nik,nkj->nij", X, X.permute(0, 2, 1))
        L_Y = torch.einsum("nik,nkj->nij", Y, Y.permute(0, 2, 1))
        return torch.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X: torch.Tensor, Y: torch.Tensor):

        # [n (num_layes), b (batch), d (dimension)]
        X = torch.nn.functional.normalize(X, dim=2)
        Y = torch.nn.functional.normalize(Y, dim=2)

        hsic = self.linear_HSIC(X, Y)  # [n, b, b]
        var1 = torch.sqrt(self.linear_HSIC(X, X) + 1e-6)  # [n, b, b]
        var2 = torch.sqrt(self.linear_HSIC(Y, Y) + 1e-6)  # [n, b, b]
        return hsic / (var1 * var2)

    def kernel_CKA(self, X, Y, sigma=None):

        assert len(X.shape) == 2, "kernel_CKA only support 2d tensor currently"
        X = torch.nn.functional.normalize(X, dim=1)
        Y = torch.nn.functional.normalize(Y, dim=1)
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = torch.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = torch.sqrt(self.kernel_HSIC(Y, Y, sigma))
        return hsic / (var1 * var2)
