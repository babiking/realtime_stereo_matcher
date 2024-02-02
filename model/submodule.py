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
