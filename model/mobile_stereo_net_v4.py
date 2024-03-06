# MobileStereoNetV2 implementation based on: https://github.com/zjjMaiMai/TinyHITNet

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_cost_volume(left, right, max_disp):
    # left: 1 x 32 x 60 x 80
    # right: 1 x 32 x 60 x 80
    # max_disp: 24
    # cost_volume: 1 x 32 x 24 x 60 x 80
    n, c, h, w = left.shape

    cost_volume = torch.ones(
        size=[n, c, max_disp, h, w], dtype=left.dtype, device=left.device
    )

    # for any disparity d:
    #   cost_volume[:, :, d, :, :d] = 1.0
    #   cost_volume[:, :, d, :, d:] = left[:, :, :, d:] - right[:, :, :, :-d]
    cost_volume[:, :, 0, :, :] = left - right
    for d in range(1, max_disp):
        cost_volume[:, :, d, :, d:] = left[:, :, :, d:] - right[:, :, :, :-d]

    # cost_volume: 1 x 32 x 24 x 60 x 80
    return cost_volume


def conv_3x3(in_c, out_c, s=1, d=1):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, s, d, dilation=d, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(),
    )


def conv_1x1(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(),
    )


class ResBlock(nn.Module):
    def __init__(self, c0, dilation=1):
        super().__init__()
        self.conv = nn.Sequential(
            conv_3x3(c0, c0, d=dilation),
            conv_3x3(c0, c0, d=dilation),
        )

    def forward(self, input):
        x = self.conv(input)
        return x + input


class RefineNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, refine_dilates):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.refine_dilates = refine_dilates
        self.conv0 = nn.Sequential(
            conv_3x3(self.in_dim, self.hidden_dim),
            *[ResBlock(self.hidden_dim, d) for d in refine_dilates],
            nn.Conv2d(self.hidden_dim, 1, 3, 1, 1),
        )

    def forward(self, disp, l_fmap):
        # disp: 1 x 1 x 60 x 80
        # l_fmap: 1 x 32 x 120 x 160
        # r_fmap: 1 x 32 x 120 x 160

        # disp: 1 x 1 x 120 x 160
        disp = F.interpolate(
            disp * 2.0, scale_factor=2, mode="bilinear", align_corners=False
        )

        # x: 1 x (2 * 32 + 1) x 120 x 160
        x = torch.cat((disp, l_fmap), dim=1)
        # x: 1 x 1 x 120 x 160
        x = self.conv0(x)
        # x: 1 x 1 x 120 x 160, x >= 0.0
        return F.relu(disp + x)


def same_padding_conv(x, w, b, s):
    out_h = math.ceil(x.size(2) / s[0])
    out_w = math.ceil(x.size(3) / s[1])

    pad_h = max((out_h - 1) * s[0] + w.size(2) - x.size(2), 0)
    pad_w = max((out_w - 1) * s[1] + w.size(3) - x.size(3), 0)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
    x = F.conv2d(x, w, b, stride=s)
    return x


class SameConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return same_padding_conv(x, self.weight, self.bias, self.stride)


class UpsampleBlock(nn.Module):
    def __init__(self, c0, c1):
        super().__init__()
        self.up_conv = nn.Sequential(
            nn.ConvTranspose2d(c0, c1, 2, 2),
            nn.LeakyReLU(0.2),
        )
        self.merge_conv = nn.Sequential(
            nn.Conv2d(c1 * 2, c1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(c1, c1, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(c1, c1, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )

    def forward(self, input, sc):
        x = self.up_conv(input)
        x = torch.cat((x, sc), dim=1)
        x = self.merge_conv(x)
        return x


class UNetFeatureExtractor(nn.Module):
    def __init__(self, hidden_dims, up_factor):
        super().__init__()

        self.hidden_dims = hidden_dims
        self.down_factor = len(hidden_dims) - 1
        self.up_factor = up_factor

        self.down_layers = nn.ModuleList([])
        self.up_layers = nn.ModuleList([])

        for i in range(self.down_factor + 1):
            if i == 0:
                layer = nn.Sequential(
                    nn.Conv2d(3, self.hidden_dims[0], 3, 1, 1),
                    nn.LeakyReLU(0.2),
                )
            elif i > 0 and i < self.down_factor:
                layer = nn.Sequential(
                    SameConv2d(self.hidden_dims[i - 1], self.hidden_dims[i], 4, 2),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(self.hidden_dims[i], self.hidden_dims[i], 3, 1, 1),
                    nn.LeakyReLU(0.2),
                )
            elif i == self.down_factor:
                layer = nn.Sequential(
                    SameConv2d(self.hidden_dims[i - 1], self.hidden_dims[i], 4, 2),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(self.hidden_dims[i], self.hidden_dims[i], 3, 1, 1),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(self.hidden_dims[i], self.hidden_dims[i], 3, 1, 1),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(self.hidden_dims[i], self.hidden_dims[i], 3, 1, 1),
                    nn.LeakyReLU(0.2),
                )
            self.down_layers.append(layer)

        for i in range(self.up_factor):
            j = self.down_factor - i

            layer = UpsampleBlock(self.hidden_dims[j], self.hidden_dims[j - 1])
            self.up_layers.append(layer)

    def forward(self, x):
        down_pyramid = []
        for i, down_layer in enumerate(self.down_layers):
            x = down_layer(x)
            down_pyramid.append(x)

        up_pyramid = []
        for i, up_layer in enumerate(self.up_layers):
            j = self.down_factor - i

            y = up_layer(
                up_pyramid[i - 1] if i > 0 else down_pyramid[-1], down_pyramid[j - 1]
            )
            up_pyramid.append(y)
        return up_pyramid


class MobileStereoNetV4(nn.Module):
    def __init__(
        self,
        hidden_dims=[16, 24, 32, 48, 64],
        up_factor=2,
        max_disp=192,
        filter_dim=32,
        refine_dim=32,
        refine_dilates=[1, 2, 4, 8],
    ):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.down_factor = len(hidden_dims) - 1
        self.up_factor = up_factor
        self.max_disp = (max_disp + 1) // (2 ** (self.down_factor - 1))
        self.filter_dim = filter_dim
        self.refine_dim = refine_dim
        self.refine_dilates = refine_dilates
        self.feature_extractor = UNetFeatureExtractor(
            hidden_dims=hidden_dims, up_factor=up_factor
        )

        self.cost_filter = nn.Sequential(
            nn.Conv3d(self.hidden_dims[self.down_factor - 1], self.filter_dim, 3, 1, 1),
            nn.BatchNorm3d(self.filter_dim),
            nn.ReLU(),
            nn.Conv3d(self.filter_dim, self.filter_dim, 3, 1, 1),
            nn.BatchNorm3d(self.filter_dim),
            nn.ReLU(),
            nn.Conv3d(self.filter_dim, self.filter_dim, 3, 1, 1),
            nn.BatchNorm3d(self.filter_dim),
            nn.ReLU(),
            nn.Conv3d(self.filter_dim, 1, 3, 1, 1),
        )
        self.refine_layers = nn.ModuleList(
            [
                RefineNet(
                    in_dim=1 + self.hidden_dims[self.down_factor - 2 - i],
                    hidden_dim=self.refine_dim,
                    refine_dilates=self.refine_dilates,
                )
                for i in range(self.up_factor - 1)
            ]
        )

    def forward(self, l_img, r_img, is_train=True):
        # l_fmaps:
        #   [1] 1 x 32 x 60 x 80, i.e. 8x downsample
        #   [2] 1 x 32 x 120 x 160
        #   [3] 1 x 32 x 240 x 320
        #   [4] 1 x 32 x 480 x 640
        l_fmaps = self.feature_extractor(l_img)
        r_fmaps = self.feature_extractor(r_img)

        # max_disp: 192 // 8 = 24
        # cost_volume: 1 x 32 x 24 x 60 x 80
        cost_volume = make_cost_volume(l_fmaps[0], r_fmaps[0], self.max_disp)
        # cost_volume: 1 x 24 x 60 x 80
        cost_volume = self.cost_filter(cost_volume).squeeze(1)

        x = F.softmax(cost_volume, dim=1)
        d = torch.arange(0, self.max_disp, device=x.device, dtype=x.dtype)
        # x: 1 x 1 x 60 x 80
        x = torch.sum(x * d.view(1, -1, 1, 1), dim=1, keepdim=True)

        multi_scale = []
        for i, refine in enumerate(self.refine_layers):
            # x: 1 x 1 x 60 x 80
            # l_fmaps[i + 1]: 1 x 32 x 120 x 160
            x = refine(x, l_fmaps[i + 1])
            # full_res: 1 x 1 x 480 x 640

            if (not is_train) and (i != len(self.refine_layers) - 1):
                continue

            multi_scale.append(x)

        multi_scale.append(
            F.interpolate(
                multi_scale[-1] * 4.0,
                scale_factor=4,
                mode="bilinear",
                align_corners=False,
            )
        )
        return multi_scale
