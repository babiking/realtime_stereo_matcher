# MobileStereoNetV2 implementation based on: https://github.com/zjjMaiMai/TinyHITNet

import math
import numpy as np
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


def make_groupwise_cost_volume(left, right, max_disp, n_groups=16):
    def groupwise(x, y, n_groups):
        n, c, h, w = left.shape

        assert (
            c % n_groups == 0
        ), f"groupwise cost channel ({c}) % #groups ({n_groups}) != 0."

        cost = (x * y).view([n, n_groups, c // n_groups, h, w]).mean(dim=2)
        return cost

    # left: 1 x 64 x 60 x 80
    # right: 1 x 64 x 60 x 80
    # max_disp: 24
    # n_groups: 16
    # cost_volume: 1 x 16 x 24 x 60 x 80
    n, c, h, w = left.shape

    volume = torch.zeros(
        [n, n_groups, max_disp, h, w], dtype=left.dtype, device=left.device
    )
    for d in range(max_disp):
        if d == 0:
            volume[:, :, d, :, :] = groupwise(left, right, n_groups)
        else:
            volume[:, :, d, :, d:] = groupwise(
                left[:, :, :, d:], right[:, :, :, :-d], n_groups
            )

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


class Difference3DCostVolume(nn.Module):
    def __init__(self, hidden_dim, max_disp, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.hidden_dim = hidden_dim
        self.max_disp = max_disp

    def forward(self, l_fmap, r_fmap):
        cost_volume = []

        for d in range(self.max_disp):
            if d == 0:
                cost_item = (l_fmap - r_fmap).unsqueeze(2)
            else:
                cost_item = l_fmap[:, :, :, d:] - r_fmap[:, :, :, :-d]
                cost_item = F.pad(cost_item, pad=(d, 0), mode="constant", value=1.0)
            cost_volume.append(cost_item)
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


class Warp1DOp(nn.Module):
    def __init__(
        self, mode="bilinear", padding_mode="zeros", align_corners=True, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners

    def get_random_inputs(self, n=1, c=32, h=30, w=40):
        image = torch.randn(size=(n, c, h, w), dtype=torch.float32)
        disparity = torch.randn(size=(n, 1, h, w), dtype=torch.float32) * w
        return (image, disparity)

    def get_output_number(self):
        return 1

    def warp_by_flow_map(self, img, flow):
        """
        warp image according to stereo flow map (i.e. disparity map)

        Args:
            [1] img, N x C x H x W, original image or feature map
            [2] flow,  N x 1 x H x W or N x 2 x H x W, flow map

        Return:
            [1] warped, N x C x H x W, warped image or feature map
        """
        n, c, h, w = flow.shape

        assert c == 1 or c == 2, f"invalid flow map dimension 1 or 2 ({c})!"

        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=img.device, dtype=img.dtype),
            torch.arange(w, device=img.device, dtype=img.dtype),
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
            img,
            grid_map,
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )
        return warped

    def process_grid_coordinates(self, pixs, width):
        # if self.align_corners:
        #     pixels = (grids + 1.0) * (0.5 * (width - 1))
        # else:
        #     pixels = (grids + 1.0) * (0.5 * width) - 0.5
        assert self.align_corners

        if self.padding_mode == "border":
            pixs = torch.clip(pixs, 0, width - 1)
        elif self.padding_mode == "zeros":
            pixs = torch.clip(pixs, -1, width) + 1
        elif self.padding_mode == "reflection":
            if self.align_corners:
                pixs = (width - 1) - torch.abs(
                    pixs % (2 * max(width - 1, 1)) - (width - 1)
                )
            else:
                pixs = width - torch.abs((pixs + 0.5) % (2 * width) - width) - 0.5
                pixs = torch.clip(pixs, 0, width - 1)
        return pixs

    def warp_by_disparity_map(self, img, disp):
        """
        reference: https://github.com/AlexanderLutsenko/nobuco/blob/aa4745e6abb1124d90f7d3ace6d282f923f08a40/nobuco/node_converters/grid_sampling.py#L38
        """
        n, _, h, w = img.shape

        x = torch.arange(w, device=img.device, dtype=img.dtype)
        x = x[None, None, None, :].repeat([1, 1, h, 1])
        x = x - disp

        x = self.process_grid_coordinates(x, w)

        if self.padding_mode == "zeros":
            x = F.pad(x, (1, 1, 0, 0), mode="constant", value=0)
            img = F.pad(img, (1, 1, 0, 0), mode="constant", value=0)

        if self.mode == "bilinear":
            x0 = torch.floor(x).type(dtype=torch.int64)
            x1 = torch.ceil(x).type(dtype=torch.int64)

            dx = x - x0

            v_x0 = torch.gather(img, dim=-1, index=x0.expand_as(img))
            v_x1 = torch.gather(img, dim=-1, index=x1.expand_as(img))

            warped = (1.0 - dx) * v_x0 + dx * v_x1

        elif self.mode == "nearest":
            x0 = torch.round(x).type(dtype=torch.int64)
            warped = torch.gather(img, dim=-1, index=x0.expand_as(img))
        else:
            raise NotImplementedError

        if self.padding_mode == "zeros":
            warped = warped[:, :, :, 1:-1]
        return warped

    def forward(self, img, disp):
        return self.warp_by_disparity_map(img, disp)


class RefineNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, refine_dilates, use_warp_feature):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.refine_dilates = refine_dilates
        self.use_warp_feature = use_warp_feature
        self.conv0 = nn.Sequential(
            conv_3x3(self.in_dim, self.hidden_dim),
            *[ResBlock(self.hidden_dim, d) for d in refine_dilates],
            nn.Conv2d(self.hidden_dim, 1, 3, 1, 1),
        )
        self.warp_head = Warp1DOp(
            mode="nearest", padding_mode="border", align_corners=True
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
        # rgb: 1 x 3 x 120 x 160
        if self.use_warp_feature:
            r_fmap = self.warp_head(r_fmap, disp)

            # x: 1 x (2 * 32 + 1) x 120 x 160
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


class MobileStereoNetV3(nn.Module):
    def __init__(
        self,
        down_factor=3,
        max_disp=192,
        refine_dilates=[1, 2, 4, 8, 1, 1],
        hidden_dim=32,
        use_groupwise_cost=True,
        num_groups=8,
        use_conv2d_cost=True,
        use_warp_feature=True,
        early_stop=2,
    ):
        super().__init__()
        self.down_factor = down_factor
        self.align = 2**self.down_factor
        self.max_disp = (max_disp + 1) // (2**self.down_factor)

        self.early_stop = early_stop

        self.refine_dilates = refine_dilates
        self.hidden_dim = hidden_dim

        self.use_groupwise_cost = use_groupwise_cost
        self.num_groups = num_groups

        self.use_conv2d_cost = use_conv2d_cost
        self.use_warp_feature = use_warp_feature

        self.feature_extractor = UNetFeatureExtractor(
            hidden_dims=[hidden_dim] * (down_factor + 1)
        )
        self.cost_volume = Difference3DCostVolume(
            hidden_dim=hidden_dim, max_disp=self.max_disp
        )
        self.cost_filter = nn.Sequential(
            nn.Conv3d(
                (self.num_groups if self.use_groupwise_cost else self.hidden_dim),
                self.hidden_dim,
                3,
                1,
                1,
            ),
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
        self.refine_layers = nn.ModuleList(
            [
                RefineNet(
                    in_dim=1 + (2 if self.use_warp_feature else 1) * hidden_dim,
                    hidden_dim=self.hidden_dim,
                    refine_dilates=self.refine_dilates,
                    use_warp_feature=self.use_warp_feature,
                )
                for _ in range(self.down_factor - self.early_stop)
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
        if self.use_groupwise_cost:
            cost_volume = make_groupwise_cost_volume(
                l_fmaps[0], r_fmaps[0], self.max_disp, self.num_groups
            )
        else:
            cost_volume = (
                self.cost_volume(l_fmaps[0], r_fmaps[0])
                if self.use_conv2d_cost
                else make_cost_volume(l_fmaps[0], r_fmaps[0], self.max_disp)
            )
        # cost_volume: 1 x 24 x 60 x 80
        cost_volume = self.cost_filter(cost_volume).squeeze(1)

        x = F.softmax(cost_volume, dim=1)
        d = torch.arange(0, self.max_disp, device=x.device, dtype=x.dtype)
        # x: 1 x 1 x 60 x 80
        x = torch.sum(x * d.view(1, -1, 1, 1), dim=1, keepdim=True)

        multi_scale = [x] if is_train else []
        for i, refine in enumerate(self.refine_layers):
            # x: 1 x 1 x 60 x 80
            # l_fmaps[i + 1]: 1 x 32 x 120 x 160
            x = refine(x, l_fmaps[i + 1], r_fmaps[i + 1])

            if is_train or i == len(self.refine_layers) - 1:
                multi_scale.append(x)

        if self.early_stop > 0:
            scale = float(l_img.shape[-1]) / multi_scale[-1].shape[-1]

            disp_final = (
                F.interpolate(
                    multi_scale[-1],
                    size=(l_img.shape[-2:]),
                    mode="bilinear",
                    align_corners=False,
                )
                * scale
            )
            multi_scale.append(disp_final)

        if is_train:
            return multi_scale, [None] * len(multi_scale), [None] * len(multi_scale)
        else:
            return [multi_scale[-1]]
