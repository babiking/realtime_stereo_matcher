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

    cost_volume = torch.ones(size=[n, c, max_disp, h, w],
                             dtype=left.dtype,
                             device=left.device)

    # for any disparity d:
    #   cost_volume[:, :, d, :, :d] = 1.0
    #   cost_volume[:, :, d, :, d:] = left[:, :, :, d:] - right[:, :, :, :-d]
    cost_volume[:, :, 0, :, :] = left - right
    for d in range(1, max_disp):
        cost_volume[:, :, d, :, d:] = left[:, :, :, d:] - right[:, :, :, :-d]

    # cost_volume: 1 x 32 x 24 x 60 x 80
    return cost_volume


def make_groupwise_cost_volume(left, right, max_disp, n_groups=16):

    def groupwise(x, y, n_groups):
        n, c, h, w = left.shape

        assert (c % n_groups == 0), \
            f"groupwise cost channel ({c}) % #groups ({n_groups}) != 0."

        cost = (x * y).view([n, n_groups, c // n_groups, h, w]).mean(dim=2)
        return cost

    # left: 1 x 64 x 60 x 80
    # right: 1 x 64 x 60 x 80
    # max_disp: 24
    # n_groups: 16
    # cost_volume: 1 x 16 x 24 x 60 x 80
    n, c, h, w = left.shape

    volume = torch.zeros([n, n_groups, max_disp, h, w],
                         dtype=left.dtype,
                         device=left.device)
    for d in range(max_disp):
        if d == 0:
            volume[:, :, d, :, :] = groupwise(left, right, n_groups)
        else:
            volume[:, :, d, :, d:] = \
                groupwise(left[:, :, :, d:], right[:, :, :, :-d], n_groups)

        volume = volume.contiguous()
        return volume


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


def warp_by_flow_map(image, flow):
    """
    warp image according to stereo flow map (i.e. disparity map)

    Args:
        [1] image, N x C x H x W, original image or feature map
        [2] flow,  N x 1 x H x W or N x 2 x H x W, flow map

    Return:
        [1] warped, N x C x H x W, warped image or feature map
    """
    n, c, h, w = flow.shape

    assert c == 1 or c == 2, f"invalid flow map dimension 1 or 2 ({c})!"

    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, device=image.device, dtype=image.dtype),
        torch.arange(w, device=image.device, dtype=image.dtype),
        indexing="ij",
    )

    grid_x = grid_x.view([1, 1, h, w]) - flow[:, 0, :, :].view([n, 1, h, w])
    grid_x = grid_x.permute([0, 2, 3, 1])

    if c == 2:
        grid_y = grid_y.view([1, 1, h, w]) - flow[:, 1, :, :].view(
            [n, 1, h, w])
        grid_y = grid_y.permute([0, 2, 3, 1])
    else:
        grid_y = grid_y.view([1, h, w, 1]).repeat(n, 1, 1, 1)

    grid_x = 2.0 * grid_x / (w - 1.0) - 1.0
    grid_y = 2.0 * grid_y / (h - 1.0) - 1.0
    grid_map = torch.concatenate((grid_x, grid_y), dim=-1)

    warped = F.grid_sample(image,
                           grid_map,
                           mode="bilinear",
                           padding_mode="zeros",
                           align_corners=True)
    return warped


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

    def forward(self, disp, l_fmap, r_fmap):
        # disp: 1 x 1 x 60 x 80
        # l_fmap: 1 x 32 x 120 x 160
        # r_fmap: 1 x 32 x 120 x 160

        # disp: 1 x 1 x 120 x 160
        disp = (F.interpolate(
            disp, scale_factor=2, mode="bilinear", align_corners=False) * 2)
        # rgb: 1 x 3 x 120 x 160
        if l_fmap.shape[2:] != disp.shape[2:] or r_fmap.shape[
                2:] != disp.shape[2:]:
            l_fmap = F.interpolate(
                l_fmap,
                (disp.size(2), disp.size(3)),
                mode="bilinear",
                align_corners=False,
            )
            r_fmap = F.interpolate(
                r_fmap,
                (disp.size(2), disp.size(3)),
                mode="bilinear",
                align_corners=False,
            )
        r_fmap = warp_by_flow_map(r_fmap, disp)

        # x: 1 x (2 * 32 + 1) x 120 x 160
        x = torch.cat((disp, l_fmap, r_fmap), dim=1)
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

    def __init__(self, hidden_dims):
        super().__init__()

        self.hidden_dims = hidden_dims
        self.down_factor = len(hidden_dims) - 1
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
                    SameConv2d(self.hidden_dims[i - 1], self.hidden_dims[i], 4,
                               2),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(self.hidden_dims[i], self.hidden_dims[i], 3, 1,
                              1),
                    nn.LeakyReLU(0.2),
                )
            elif i == self.down_factor:
                layer = nn.Sequential(
                    SameConv2d(self.hidden_dims[i - 1], self.hidden_dims[i], 4,
                               2),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(self.hidden_dims[i], self.hidden_dims[i], 3, 1,
                              1),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(self.hidden_dims[i], self.hidden_dims[i], 3, 1,
                              1),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(self.hidden_dims[i], self.hidden_dims[i], 3, 1,
                              1),
                    nn.LeakyReLU(0.2),
                )
            self.down_layers.append(layer)

        for i in range(self.down_factor):
            j = self.down_factor - i

            layer = UpsampleBlock(self.hidden_dims[j], self.hidden_dims[j - 1])
            self.up_layers.append(layer)

    def forward(self, x):
        down_pyramid = []
        for i, down_layer in enumerate(self.down_layers):
            x = down_layer(x)
            down_pyramid.append(x)

        up_pyramid = [down_pyramid[-1]]
        for i, up_layer in enumerate(self.up_layers):
            j = self.down_factor - i

            y = up_layer(up_pyramid[i], down_pyramid[j - 1])
            up_pyramid.append(y)
        return up_pyramid


class MobileStereoNetV3(nn.Module):

    def __init__(
        self,
        down_factor=3,
        max_disp=192,
        refine_dilates=[1, 2, 4, 8, 1, 1],
        hidden_dim=32,
        use_groupwise_cost=True,
        num_groups=8,
    ):
        super().__init__()
        self.down_factor = down_factor
        self.align = 2**self.down_factor
        self.max_disp = (max_disp + 1) // (2**self.down_factor)

        self.refine_dilates = refine_dilates
        self.hidden_dim = hidden_dim

        self.use_groupwise_cost = use_groupwise_cost
        self.num_groups = num_groups

        self.feature_extractor = UNetFeatureExtractor(
            hidden_dims=[hidden_dim] * (down_factor + 1))

        self.cost_filter = nn.Sequential(
            nn.Conv3d(
                (self.num_groups if self.use_groupwise_cost else self.hidden_dim),\
                self.hidden_dim, 3, 1, 1),
            nn.BatchNorm3d(self.hidden_dim),
            nn.ReLU(),
            nn.Conv3d(self.hidden_dim, self.hidden_dim, 3, 1, 1),
            nn.BatchNorm3d(self.hidden_dim),
            nn.ReLU(),
            nn.Conv3d(self.hidden_dim, self.hidden_dim, 3, 1, 1),
            nn.BatchNorm3d(self.hidden_dim),
            nn.ReLU(),
            nn.Conv3d(self.hidden_dim, self.hidden_dim, 3, 1, 1),
            nn.BatchNorm3d(self.hidden_dim),
            nn.ReLU(),
            nn.Conv3d(self.hidden_dim, 1, 3, 1, 1),
        )
        self.refine_layers = nn.ModuleList([
            RefineNet(
                in_dim=1 + 2 * hidden_dim,
                hidden_dim=self.hidden_dim,
                refine_dilates=self.refine_dilates,
            ) for _ in range(self.down_factor)
        ])

    def forward(self, l_img, r_img):
        l_img = (2.0 * (l_img / 255.0) - 1.0).contiguous()
        r_img = (2.0 * (r_img / 255.0) - 1.0).contiguous()

        n, c, h, w = l_img.size()
        w_pad = (self.align - (w % self.align)) % self.align
        h_pad = (self.align - (h % self.align)) % self.align

        # l_img: 1 x 3 x 480 x 640
        l_img = F.pad(l_img, (0, w_pad, 0, h_pad))
        r_img = F.pad(r_img, (0, w_pad, 0, h_pad))

        # l_fmaps:
        #   [1] 1 x 32 x 60 x 80, i.e. 8x downsample
        #   [2] 1 x 32 x 120 x 160
        #   [3] 1 x 32 x 240 x 320
        #   [4] 1 x 32 x 480 x 640
        l_fmaps = self.feature_extractor(l_img)
        r_fmaps = self.feature_extractor(r_img)

        # max_disp: 192 // 8 = 24
        # cost_volume: 1 x 32 x 24 x 60 x 80
        cost_volume = make_groupwise_cost_volume(l_fmaps[0], r_fmaps[0], self.max_disp, self.num_groups) \
            if self.use_groupwise_cost else make_cost_volume(l_fmaps[0], r_fmaps[0], self.max_disp)
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
            x = refine(x, l_fmaps[i + 1], r_fmaps[i + 1])
            scale = l_img.size(3) / x.size(3)
            # full_res: 1 x 1 x 480 x 640
            full_res = F.interpolate(x * scale,
                                     (l_img.shape[2:]))[:, :, :h, :w]
            multi_scale.append(full_res)

        return multi_scale
