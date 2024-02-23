import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SameConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def same_padding_conv(self, x, w, b, s):
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

    def forward(self, x):
        return self.same_padding_conv(x, self.weight, self.bias, self.stride)


class UpMergeConvT2d(nn.Module):
    def __init__(self, in_dim, out_dim, cat_dim=0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.cat_dim = cat_dim

        self.up_conv = nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, 2, 2),
            nn.LeakyReLU(0.2),
        )
        self.merge_conv = nn.Sequential(
            nn.Conv2d((out_dim + cat_dim), out_dim, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_dim, out_dim, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_dim, out_dim, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )

    def forward(self, input, merge=None):
        x = self.up_conv(input)
        if merge is not None and merge.shape[1] == self.cat_dim:
            x = torch.concat((x, merge), dim=1)
            x = self.merge_conv(x)
        return x


class WarpHead(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, img, flow):
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
            img, grid_map, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        return warped


def get_norm2d(out_dim, norm_type, channels_per_group=8):
    if norm_type == "group":
        return nn.GroupNorm(
            num_groups=(out_dim // channels_per_group), num_channels=out_dim
        )
    elif norm_type == "batch":
        return nn.BatchNorm2d(out_dim)
    elif norm_type == "instance":
        return nn.InstanceNorm2d(out_dim)
    elif norm_type == "none":
        return nn.Sequential()
    else:
        raise NotImplementedError(f"invalid norm type: {norm_type}!")


def get_conv2d_3x3(
    in_dim, out_dim, stride, dilation, norm_type, channels_per_group=8, use_relu=True
):
    conv = nn.Conv2d(
        in_dim,
        out_dim,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
    )
    norm = get_norm2d(out_dim, norm_type, channels_per_group)
    relu = nn.ReLU() if use_relu else nn.Sequential()
    return nn.Sequential(conv, norm, relu)


def get_conv2d_1x1(
    in_dim, out_dim, stride, dilation, norm_type, channels_per_group=8, use_relu=True
):
    conv = nn.Conv2d(
        in_dim, out_dim, kernel_size=1, stride=stride, padding=0, dilation=dilation
    )
    norm = get_norm2d(out_dim, norm_type, channels_per_group)
    relu = nn.ReLU() if use_relu else nn.Sequential()
    return nn.Sequential(conv, norm, relu)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        norm_type="group",
        stride=1,
        dilation=1,
        channels_per_group=8,
    ):
        super(ResidualBlock, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.norm_type = norm_type
        self.stride = stride
        self.dilation = dilation
        self.channels_per_group = channels_per_group

        self.conv1 = get_conv2d_3x3(
            in_dim=in_dim,
            out_dim=out_dim,
            stride=stride,
            dilation=dilation,
            norm_type=norm_type,
            channels_per_group=channels_per_group,
            use_relu=True,
        )
        self.conv2 = get_conv2d_3x3(
            in_dim=out_dim,
            out_dim=out_dim,
            stride=1,
            dilation=dilation,
            norm_type=norm_type,
            channels_per_group=channels_per_group,
            use_relu=True,
        )

        if stride == 1 and in_dim == out_dim:
            self.downsample = None
        else:
            self.downsample = get_conv2d_1x1(
                in_dim=in_dim,
                out_dim=out_dim,
                stride=stride,
                dilation=dilation,
                norm_type=norm_type,
                channels_per_group=channels_per_group,
                use_relu=False,
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        y = x
        y = self.conv1(y)
        y = self.conv2(y)

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class MobileResidualBlockV2(nn.Module):
    def __init__(self, in_dim, out_dim, stride, expanse_ratio=1, dilation=1):
        super(MobileResidualBlockV2, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.stride = stride
        assert stride in [1, 2]
        self.expanse_ratio = expanse_ratio
        self.dilation = dilation

        self.hidden_dim = int(in_dim * expanse_ratio)

        if self.stride != 1 or self.in_dim != self.out_dim:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_dim, self.out_dim, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(self.out_dim),
            )
        else:
            self.downsample = None

        # nn.Conv2D output dimension:
        #   H_out = floor( (H_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
        # Conv2D:
        #   K x K x C_in x C_out
        # Depthwise Separable Conv2D:
        #   K x K x C_in + 1 x 1 x C_in x C_out
        if self.expanse_ratio == 1:
            self.conv = nn.Sequential(
                # depthwise
                nn.Conv2d(
                    in_channels=self.in_dim,
                    out_channels=self.hidden_dim,
                    kernel_size=3,
                    stride=stride,
                    padding=self.dilation,
                    dilation=self.dilation,
                    groups=self.hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm2d(self.hidden_dim),
                nn.ReLU6(inplace=True),
                # pointwisze-linear
                nn.Conv2d(
                    in_channels=self.hidden_dim,
                    out_channels=self.out_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(self.out_dim),
            )
        else:
            self.conv = nn.Sequential(
                # pointwise
                nn.Conv2d(
                    in_channels=self.in_dim,
                    out_channels=self.hidden_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(self.hidden_dim),
                nn.ReLU6(inplace=True),
                # depthwise
                nn.Conv2d(
                    in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim,
                    kernel_size=3,
                    stride=stride,
                    padding=self.dilation,
                    dilation=self.dilation,
                    groups=self.hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm2d(self.hidden_dim),
                nn.ReLU6(inplace=True),
                # pointwise-linear
                nn.Conv2d(
                    in_channels=self.hidden_dim,
                    out_channels=self.out_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(self.out_dim),
            )

    def forward(self, x):
        return self.conv(x) + (x if self.downsample is None else self.downsample(x))
