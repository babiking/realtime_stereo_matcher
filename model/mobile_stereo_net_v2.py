# MobileStereoNetV2 implementation based on: https://github.com/zjjMaiMai/TinyHITNet

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

    def forward(self, disp, l_rgb, r_rgb):
        # disp: 1 x 1 x 60 x 80
        # rgb: 1 x 3 x 480 x 640

        # disp: 1 x 1 x 120 x 160
        disp = (
            F.interpolate(disp, scale_factor=2, mode="bilinear", align_corners=False)
            * 2
        )
        # rgb: 1 x 3 x 120 x 160
        l_rgb = F.interpolate(
            l_rgb, (disp.size(2), disp.size(3)), mode="bilinear", align_corners=False
        )
        r_rgb = F.interpolate(
            r_rgb, (disp.size(2), disp.size(3)), mode="bilinear", align_corners=False
        )
        r_rgb = warp_by_flow_map(r_rgb, disp)

        # x: 1 x 4 x 120 x 160
        x = torch.cat((disp, l_rgb, r_rgb), dim=1)
        # x: 1 x 1 x 120 x 160
        x = self.conv0(x)
        # x: 1 x 1 x 120 x 160, x >= 0.0
        return F.relu(disp + x)


class MobileStereoNetV2(nn.Module):
    def __init__(
        self,
        down_factor=3,
        max_disp=192,
        refine_dim=7,
        refine_dilates=[1, 2, 4, 8, 1, 1],
        hidden_dim=32,
    ):
        super().__init__()
        self.down_factor = down_factor
        self.align = 2**self.down_factor
        self.max_disp = (max_disp + 1) // (2**self.down_factor)

        self.refine_dim = refine_dim
        self.refine_dilates = refine_dilates
        self.hidden_dim = hidden_dim

        self.feature_extractor = [
            conv_3x3(3, self.hidden_dim, 2),
            ResBlock(self.hidden_dim),
        ]
        for _ in range(self.down_factor - 1):
            self.feature_extractor += [
                conv_3x3(self.hidden_dim, self.hidden_dim, 2),
                ResBlock(self.hidden_dim),
            ]
        self.feature_extractor += [nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, 1, 1)]
        self.feature_extractor = nn.Sequential(*self.feature_extractor)

        self.cost_filter = nn.Sequential(
            nn.Conv3d(self.hidden_dim, self.hidden_dim, 3, 1, 1),
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
        self.refine_layer = nn.ModuleList(
            [
                RefineNet(
                    in_dim=self.refine_dim,
                    hidden_dim=self.hidden_dim,
                    refine_dilates=self.refine_dilates,
                )
                for _ in range(self.down_factor)
            ]
        )

    def forward(self, left_img, right_img):
        left_img = (2.0 * (left_img / 255.0) - 1.0).contiguous()
        right_img = (2.0 * (right_img / 255.0) - 1.0).contiguous()

        n, c, h, w = left_img.size()
        w_pad = (self.align - (w % self.align)) % self.align
        h_pad = (self.align - (h % self.align)) % self.align

        # left_img: 1 x 3 x 480 x 640
        left_img = F.pad(left_img, (0, w_pad, 0, h_pad))
        right_img = F.pad(right_img, (0, w_pad, 0, h_pad))

        # lf: 1 x 32 x 60 x 80, i.e. 8x downsample
        lf = self.feature_extractor(left_img)
        rf = self.feature_extractor(right_img)

        # lf: 1 x 32 x 60 x 80
        # rf: 1 x 32 x 60 x 80
        # max_disp: 192 // 8 = 24
        # cost_volume: 1 x 32 x 24 x 60 x 80
        cost_volume = make_cost_volume(lf, rf, self.max_disp)
        # cost_volume: 1 x 24 x 60 x 80
        cost_volume = self.cost_filter(cost_volume).squeeze(1)

        x = F.softmax(cost_volume, dim=1)
        d = torch.arange(0, self.max_disp, device=x.device, dtype=x.dtype)
        # x: 1 x 1 x 60 x 80
        x = torch.sum(x * d.view(1, -1, 1, 1), dim=1, keepdim=True)

        multi_scale = []
        for refine in self.refine_layer:
            # x: 1 x 1 x 60 x 80
            # left_img: 1 x 3 x 480 x 640
            x = refine(x, left_img, right_img)
            scale = left_img.size(3) / x.size(3)
            # full_res: 1 x 1 x 480 x 640
            full_res = F.interpolate(x * scale, left_img.shape[2:])[:, :, :h, :w]
            multi_scale.append(full_res)

        return [-1.0 * flow_map for flow_map in multi_scale]
