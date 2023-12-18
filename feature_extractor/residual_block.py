import torch
import torch.nn as nn


class MobileV2Residual(nn.Module):
    # reference: https://github.com/cogsys-tuebingen/mobilestereonet

    def __init__(self, in_dim=64, out_dim=128, stride=1, expanse_ratio=1, dilation=1):
        super(MobileV2Residual, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.stride = stride
        self.expanse_ratio = expanse_ratio
        self.dilation = dilation

        assert stride in [1, 2]

        hidden_dim = int(in_dim * expanse_ratio)
        self.use_res_connect = self.stride == 1 and in_dim == out_dim
        pad = dilation

        if expanse_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    3,
                    stride,
                    pad,
                    dilation=dilation,
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, out_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_dim),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(in_dim, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    3,
                    stride,
                    pad,
                    dilation=dilation,
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, out_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_dim),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

    def __str__(self):
        return f"{self.__class__.__name__} | in={self.in_dim} | out={self.out_dim} | stride={self.stride} | dilate={self.dilation}"


class ResidualBottleneckBlock(nn.Module):
    # reference: https://github.com/princeton-vl/RAFT-Stereo

    def __init__(self, in_dim=64, out_dim=128, norm_fn="group", stride=1):
        super(ResidualBottleneckBlock, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.norm_fn = norm_fn
        self.stride = stride

        self.conv1 = nn.Conv2d(in_dim, out_dim // 4, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(
            out_dim // 4, out_dim // 4, kernel_size=3, padding=1, stride=stride
        )
        self.conv3 = nn.Conv2d(out_dim // 4, out_dim, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

        if norm_fn == "group":
            num_groups = out_dim // 8

            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=out_dim // 4)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_dim // 4)
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=out_dim)
            if not stride == 1:
                self.norm4 = nn.GroupNorm(num_groups=num_groups, num_channels=out_dim)

        elif norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(out_dim // 4)
            self.norm2 = nn.BatchNorm2d(out_dim // 4)
            self.norm3 = nn.BatchNorm2d(out_dim)
            if not stride == 1:
                self.norm4 = nn.BatchNorm2d(out_dim)

        elif norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(out_dim // 4)
            self.norm2 = nn.InstanceNorm2d(out_dim // 4)
            self.norm3 = nn.InstanceNorm2d(out_dim)
            if not stride == 1:
                self.norm4 = nn.InstanceNorm2d(out_dim)

        elif norm_fn == "none":
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()
            if not stride == 1:
                self.norm4 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=stride), self.norm4
            )

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)

    def __str__(self):
        return f"{self.__class__.__name__} | in={self.in_dim} | out={self.out_dim} | stride={self.stride} | norm={self.norm_fn}"
