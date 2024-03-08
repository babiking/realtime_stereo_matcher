# MobileStereoNetV2 implementation based on: https://github.com/zjjMaiMai/TinyHITNet

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.submodule import BasicConv, Conv2x


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


class Difference3DCostVolume(nn.Module):
    def __init__(self, hidden_dim, max_disp, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.hidden_dim = hidden_dim
        self.max_disp = max_disp

        self.l_kernels = self.get_conv2d_kernels(reverse=False)
        self.r_kernels = self.get_conv2d_kernels(reverse=True)

    def get_conv2d_kernels(self, reverse=True):
        kernels = []
        for i in range(1, self.max_disp):
            kernel = np.zeros(shape=[self.hidden_dim, 1, 1, i + 1], dtype=np.float32)
            kernel[:, 0, 0, (0 if reverse else -1)] = 1.0
            kernel = torch.tensor(kernel, dtype=torch.float32)

            kernels.append(kernel)
        return kernels

    def forward(self, l_fmap, r_fmap):
        cost_volume = []
        for d in range(self.max_disp):
            if d == 0:
                cost_volume.append((l_fmap - r_fmap).unsqueeze(2))
            else:
                x = F.conv2d(
                    l_fmap,
                    self.l_kernels[d - 1].to(l_fmap.device),
                    stride=1,
                    padding=0,
                    groups=self.hidden_dim,
                )
                y = F.conv2d(
                    r_fmap,
                    self.r_kernels[d - 1].to(r_fmap.device),
                    stride=1,
                    padding=0,
                    groups=self.hidden_dim,
                )

                cost_volume.append(F.pad(x - y, [d, 0, 0, 0], value=1.0).unsqueeze(2))
        cost_volume = torch.concat(cost_volume, dim=2)
        return cost_volume


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
    def __init__(self, hidden_dim, refine_dilates, use_warp_feature=True):
        super().__init__()
        self.in_dim = 1 + hidden_dim * (2 if use_warp_feature else 1)
        self.hidden_dim = hidden_dim
        self.refine_dilates = refine_dilates
        self.use_warp_feature = use_warp_feature
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
        disp = (
            F.interpolate(disp, scale_factor=2, mode="bilinear", align_corners=False)
            * 2
        )

        if self.use_warp_feature:
            r_fmap = warp_by_flow_map(r_fmap, disp)

            x = torch.cat((disp, l_fmap, r_fmap), dim=1)
        else:
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

        up_pyramid = [down_pyramid[-1]]
        for i, up_layer in enumerate(self.up_layers):
            j = self.down_factor - i

            y = up_layer(up_pyramid[i], down_pyramid[j - 1])
            up_pyramid.append(y)
        return up_pyramid


class CostVolume3D(nn.Module):
    def __init__(self, hidden_dim, max_disp, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.hidden_dim = hidden_dim
        self.max_disp = max_disp

        self.l_kernels = self.get_conv2d_kernels(reverse=False)
        self.r_kernels = self.get_conv2d_kernels(reverse=True)

    def get_conv2d_kernels(self, reverse=True):
        kernels = []
        for i in range(1, self.max_disp):
            kernel = np.zeros(shape=[self.hidden_dim, 1, 1, i + 1], dtype=np.float32)
            kernel[:, 0, 0, (0 if reverse else -1)] = 1.0
            kernel = torch.tensor(kernel, dtype=torch.float32)

            kernels.append(kernel)
        return kernels

    def get_cost_item(self, l_fmap, r_fmap):
        return l_fmap - r_fmap

    def make_cost_volume_naive(self, l_fmap, r_fmap, max_disp):
        # left: 1 x 32 x 60 x 80
        # right: 1 x 32 x 60 x 80
        # max_disp: 24
        # cost_volume: 1 x 32 x 24 x 60 x 80
        n, c, h, w = l_fmap.shape

        cost_volume = torch.ones(
            size=[n, c, max_disp, h, w], dtype=l_fmap.dtype, device=l_fmap.device
        )

        # for any disparity d:
        #   cost_volume[:, :, d, :, :d] = 1.0
        #   cost_volume[:, :, d, :, d:] = left[:, :, :, d:] - right[:, :, :, :-d]
        cost_volume[:, :, 0, :, :] = self.get_cost_item(l_fmap, r_fmap)
        for d in range(1, max_disp):
            cost_volume[:, :, d, :, d:] = self.get_cost_item(
                l_fmap[:, :, :, d:], r_fmap[:, :, :, :-d]
            )

        # cost_volume: 1 x 32 x 24 x 60 x 80
        return cost_volume

    def make_cost_volume_conv2d(self, l_fmap, r_fmap, max_disp):
        cost_volume = []
        for d in range(max_disp):
            if d == 0:
                cost_volume.append(self.get_cost_item(l_fmap, r_fmap).unsqueeze(2))
            else:
                x = F.conv2d(
                    l_fmap,
                    self.l_kernels[d - 1].to(l_fmap.device),
                    stride=1,
                    padding=0,
                    groups=self.hidden_dim,
                )
                y = F.conv2d(
                    r_fmap,
                    self.r_kernels[d - 1].to(r_fmap.device),
                    stride=1,
                    padding=0,
                    groups=self.hidden_dim,
                )
                cost_item = self.get_cost_item(x, y)

                cost_volume.append(
                    F.pad(cost_item, [d, 0, 0, 0], value=1.0).unsqueeze(2)
                )
        cost_volume = torch.concat(cost_volume, dim=2)
        return cost_volume

    def forward(self, l_fmap, r_fmap, use_naive=True):
        return (
            self.make_cost_volume_naive(l_fmap, r_fmap, self.max_disp)
            if use_naive
            else self.make_cost_volume_conv2d(l_fmap, r_fmap, self.max_disp)
        )


class CostFilter3D(nn.Module):
    def __init__(self, in_dim, filter_dims, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.in_dim = in_dim
        self.filter_dims = filter_dims

        cost_filter = nn.ModuleList([])
        for i in range(len(filter_dims)):
            cost_filter.append(
                nn.Conv3d(
                    (in_dim if i == 0 else filter_dims[i - 1]), filter_dims[i], 3, 1, 1
                )
            )
            cost_filter.append(nn.BatchNorm3d(filter_dims[i]))
            cost_filter.append(nn.ReLU())
        cost_filter.append(nn.Conv3d(filter_dims[-1], 1, 3, 1, 1))
        self.cost_filter = nn.Sequential(*cost_filter)

    def forward(self, cost_volume):
        return self.cost_filter(cost_volume).squeeze(1)


class CostAttention3D(nn.Module):
    def __init__(self, img_dim, cost_dim, hidden_dim=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.img_dim = img_dim
        self.cost_dim = cost_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else (img_dim // 2)

        self.img_att = nn.Sequential(
            BasicConv(img_dim, img_dim // 2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(img_dim // 2, cost_dim, 1),
        )

    def forward(self, cost_volume, l_fmap):
        l_img_att = self.img_att(l_fmap).unsqueeze(2)
        return torch.sigmoid(l_img_att) * cost_volume


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
        grid_y = grid_y.view([1, 1, h, w]) - flow[:, 1, :, :].view([n, 1, h, w])
        grid_y = grid_y.permute([0, 2, 3, 1])
    else:
        grid_y = grid_y.view([1, h, w, 1]).repeat(n, 1, 1, 1)

    grid_x = 2.0 * grid_x / (w - 1.0) - 1.0
    grid_y = 2.0 * grid_y / (h - 1.0) - 1.0
    grid_map = torch.concatenate((grid_x, grid_y), dim=-1)

    warped = F.grid_sample(
        image, grid_map, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    return warped


def disparity_regress(cost_volume, max_disp):
    cost_probs = F.softmax(cost_volume, dim=1)
    disp_range = torch.arange(
        0, max_disp, device=cost_probs.device, dtype=cost_probs.dtype
    )
    # x: 1 x 1 x 60 x 80
    disp_map = torch.sum(cost_probs * disp_range.view(1, -1, 1, 1), dim=1, keepdim=True)
    return disp_map, cost_probs


class ContextUpsample(nn.Module):
    def __init__(
        self, in_dims, hidden_dims, scale_factor=4, unfold_radius=3, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        assert len(in_dims) == len(hidden_dims)
        assert 2 ** (len(hidden_dims) - 1) == scale_factor
        self.in_dims = in_dims
        self.hidden_dims = hidden_dims
        self.scale_factor = scale_factor
        self.unfold_radius = unfold_radius

        self.conv_ups = nn.ModuleList([])
        for i in range(len(in_dims)):
            conv_layer = nn.ModuleList([])

            conv_layer.append(
                BasicConv(
                    in_dims[0] if i == 0 else in_dims[i] + hidden_dims[i - 1],
                    hidden_dims[i],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            conv_layer.append(
                nn.Conv2d(hidden_dims[i], hidden_dims[i], 3, 1, 1, bias=False)
            )
            conv_layer.append(nn.BatchNorm2d(hidden_dims[i]))
            conv_layer.append(nn.ReLU())
            if i < len(in_dims) - 1:
                conv_layer.append(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    )
                )
                conv_layer.append(nn.BatchNorm2d(hidden_dims[i]))
                conv_layer.append(nn.ReLU())
            self.conv_ups.append(nn.Sequential(*conv_layer))

        self.conv_out = nn.Conv2d(
            hidden_dims[-1], unfold_radius**2, kernel_size=3, stride=1, padding=1
        )

    def forward(self, disp_up, l_fmaps):
        disp_weights = None
        for i in range(len(self.conv_ups)):
            disp_weights = self.conv_ups[i](
                l_fmaps[i] if i == 0 else torch.cat((disp_weights, l_fmaps[i]), dim=1)
            )
        disp_weights = self.conv_out(disp_weights)
        disp_weights = torch.softmax(disp_weights, dim=1)

        n, c, h, w = disp_up.shape

        disp_unfold = F.unfold(disp_up, self.unfold_radius, 1, 1).reshape(
            n, self.unfold_radius**2, h, w
        )
        disp_unfold = F.interpolate(
            disp_unfold,
            scale_factor=self.scale_factor,
            mode="bilinear",
            align_corners=True,
        ) * float(self.scale_factor)

        disp_up = torch.sum(disp_unfold * disp_weights, dim=1, keepdim=True)
        return disp_up


class MobileStereoNetV5(nn.Module):
    def __init__(
        self,
        extract_dims=[16, 24, 48, 64],
        up_factor=1,
        max_disp=192,
        filter_dims=[32, 32, 32, 32],
        refine_dilates=[1, 2, 4, 8, 1, 1],
        use_warp_feature=True,
        context_dims=[32, 32, 32],
    ):
        super().__init__()
        self.up_factor = up_factor

        self.down_factor = len(extract_dims) - 1
        self.max_disp = (max_disp + 1) // (2**self.down_factor)

        self.extract_dims = extract_dims
        self.feature_extractor = UNetFeatureExtractor(
            extract_dims, up_factor=len(extract_dims) - 1
        )

        self.cost_builder = CostVolume3D(
            hidden_dim=self.extract_dims[-1], max_disp=self.max_disp
        )
        self.cost_attention = CostAttention3D(
            img_dim=self.extract_dims[-1],
            cost_dim=self.extract_dims[-1],
            hidden_dim=None,
        )

        self.filter_dims = filter_dims
        self.cost_filter = CostFilter3D(
            in_dim=self.extract_dims[-1], filter_dims=filter_dims
        )

        self.refine_dilates = refine_dilates
        self.use_warp_feature = use_warp_feature
        self.refine_layers = nn.ModuleList(
            [
                RefineNet(
                    hidden_dim=self.extract_dims[self.down_factor - 1 - i],
                    refine_dilates=self.refine_dilates,
                    use_warp_feature=self.use_warp_feature,
                )
                for i in range(self.up_factor)
            ]
        )

        self.context_dims = context_dims
        self.context_upsampler = ContextUpsample(
            in_dims=np.flip(extract_dims[0:-up_factor]),
            hidden_dims=context_dims,
            scale_factor=2 ** (self.down_factor - self.up_factor),
            unfold_radius=3,
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
        cost_volume = self.cost_builder(l_fmaps[0], r_fmaps[0], use_naive=is_train)
        cost_volume = self.cost_attention(cost_volume, l_fmaps[0])

        # cost_volume: 1 x 24 x 60 x 80
        cost_volume = self.cost_filter(cost_volume)

        disp_up, _ = disparity_regress(cost_volume, max_disp=self.max_disp)

        multi_scale = [disp_up] if is_train else []
        for i, refine in enumerate(self.refine_layers):
            # x: 1 x 1 x 60 x 80
            # l_fmaps[i + 1]: 1 x 32 x 120 x 160
            disp_up = refine(disp_up, l_fmaps[i + 1], r_fmaps[i + 1])

            if is_train:
                multi_scale.append(disp_up)

        disp_up = self.context_upsampler(disp_up, l_fmaps[self.up_factor :])
        multi_scale.append(disp_up)

        if is_train:
            l_fmaps = l_fmaps[0 : self.up_factor + 1] + [l_fmaps[-1]]
            r_fmaps = r_fmaps[0 : self.up_factor + 1] + [r_fmaps[-1]]
            return (multi_scale, l_fmaps, r_fmaps)
        else:
            return multi_scale
