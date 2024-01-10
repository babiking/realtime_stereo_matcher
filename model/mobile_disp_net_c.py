# MobileDispNetC implementation based on: https://github.com/CVLAB-Unibo/Real-time-self-adaptive-deep-stereo

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal


class Conv2dBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size=3,
        stride=1,
        with_batch_norm=True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.with_batch_norm = with_batch_norm

        if with_batch_norm:
            self.layer = nn.Sequential(
                nn.Conv2d(
                    in_dim,
                    out_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=(kernel_size - 1) // 2,
                    bias=False,
                ),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.1, inplace=True),
            )
        else:
            self.layer = nn.Sequential(
                nn.Conv2d(
                    in_dim,
                    out_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=(kernel_size - 1) // 2,
                    bias=False,
                ),
                nn.LeakyReLU(0.1, inplace=True),
            )

    def forward(self, x):
        return self.layer(x)


class Conv2dTransposeBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size=3,
        stride=1,
        with_batch_norm=True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.with_batch_norm = with_batch_norm

        if with_batch_norm:
            self.layer = nn.Sequential(
                # H_out = (H_in - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_size[0] - 1) + output_padding[0] + 1
                nn.ConvTranspose2d(
                    in_dim,
                    out_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=(kernel_size - 1) // 2,
                    output_padding=stride - 1 - int(kernel_size % 2 == 0),
                    bias=False,
                    padding_mode="zeros",
                ),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.1, inplace=True),
            )
        else:
            self.layer = nn.Sequential(
                nn.ConvTranspose2d(
                    in_dim,
                    out_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=(kernel_size - 1) // 2,
                    output_padding=stride - 1 - int(kernel_size % 2 == 0),
                    bias=False,
                    padding_mode="zeros",
                ),
                nn.LeakyReLU(0.1, inplace=True),
            )

    def forward(self, x):
        return self.layer(x)


class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_dim)

        if stride != 1 or out_dim != in_dim:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_dim),
            )
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out


class UpsampleBlock(nn.Module):
    def __init__(
        self, in_dim, skip_dim, out_dim, with_batch_norm, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.in_dim = in_dim
        self.skip_dim = skip_dim
        self.out_dim = out_dim
        self.with_batch_norm = with_batch_norm

        self.deconv = Conv2dTransposeBlock(
            in_dim=in_dim,
            out_dim=out_dim,
            kernel_size=4,
            stride=2,
            with_batch_norm=with_batch_norm,
        )
        self.predict = nn.Conv2d(
            in_dim, 1, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.up_predict = nn.ConvTranspose2d(
            1, 1, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.concat = nn.Conv2d(
            skip_dim + out_dim + 1,
            out_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

    def forward(self, bottom_feat, skip_connect_feat):
        deconv_feat = self.deconv(bottom_feat)

        disp_map = self.predict(bottom_feat)
        disp_map_2x = self.up_predict(disp_map)

        concat_feat = torch.concat((skip_connect_feat, deconv_feat, disp_map_2x), dim=1)
        concat_feat = self.concat(concat_feat)
        return disp_map, concat_feat


def make_correlation_volume(l_fmap, r_fmap, max_disp):
    n, c, h, w = l_fmap.shape

    corr_volume = torch.zeros(
        size=[n, max_disp, h, w], dtype=l_fmap.dtype, device=r_fmap.device
    )

    for d in range(max_disp):
        if d == 0:
            corr_volume[:, 0, :, :] = (l_fmap * r_fmap).mean(dim=1)
        else:
            corr_volume[:, d, :, d:] = (
                l_fmap[:, :, :, d:] * r_fmap[:, :, :, :-d]
            ).mean(dim=1)

    # corr_volume: 1 x 40 x 60 x 80
    corr_volume = corr_volume.contiguous()
    return corr_volume


def disparity_regression(corr_volume, max_disp):
    assert len(corr_volume.shape) == 4, f"#dimensions of correlation volume != 4."
    assert (
        corr_volume.shape[1] == max_disp
    ), f"#channels of correlation volume != max_disparity ({max_disp})."

    disp_values = torch.arange(
        0, max_disp, dtype=corr_volume.dtype, device=corr_volume.device
    ).view([1, max_disp, 1, 1])

    corr_values = F.softmax(corr_volume, dim=1)
    corr_values = torch.sum(corr_values * disp_values, dim=1, keepdim=True)
    return corr_values


def disparity_interpolate(disp, shape):
    n, _, src_h, src_w = disp.shape

    dst_h, dst_w = shape

    if src_h != dst_h or src_w != dst_w:
        scale = float(dst_w) / src_w

        disp = F.interpolate(
            disp * scale, (dst_h, dst_w), mode="bilinear", align_corners=False
        )
    return disp


class MobileDispNetC(nn.Module):
    def __init__(self, hidden_dim=32, max_disp=192, with_batch_norm=True):
        super(MobileDispNetC, self).__init__()

        self.down_factor = 6
        self.hidden_dim = hidden_dim
        self.max_disp = max_disp
        self.with_batch_norm = with_batch_norm

        self.conv1 = Conv2dBlock(
            in_dim=3,
            out_dim=hidden_dim * (2**0),
            kernel_size=7,
            stride=2,
            with_batch_norm=with_batch_norm,
        )
        self.conv2 = Conv2dBlock(
            in_dim=hidden_dim * (2**0),
            out_dim=hidden_dim * (2**1),
            kernel_size=5,
            stride=2,
            with_batch_norm=with_batch_norm,
        )

        self.conv_redir = Conv2dBlock(
            in_dim=hidden_dim * (2**1),
            out_dim=hidden_dim * (2**0),
            kernel_size=1,
            stride=1,
            with_batch_norm=with_batch_norm,
        )

        self.conv3 = nn.Sequential(
            Conv2dBlock(
                in_dim=hidden_dim * (2**0) + (self.max_disp // 4),
                out_dim=hidden_dim * (2**2),
                kernel_size=5,
                stride=2,
                with_batch_norm=with_batch_norm,
            ),
            Conv2dBlock(
                in_dim=hidden_dim * (2**2),
                out_dim=hidden_dim * (2**2),
                kernel_size=3,
                stride=1,
                with_batch_norm=False,
            ),
        )
        self.res4 = ResBlock(hidden_dim * (2**2), hidden_dim * (2**3), stride=2)
        self.res5 = ResBlock(hidden_dim * (2**3), hidden_dim * (2**4), stride=2)
        self.res6 = ResBlock(hidden_dim * (2**4), hidden_dim * (2**5), stride=2)

        self.up5 = UpsampleBlock(
            in_dim=hidden_dim * (2**5),
            skip_dim=hidden_dim * (2**4),
            out_dim=hidden_dim * (2**4),
            with_batch_norm=with_batch_norm,
        )
        self.up4 = UpsampleBlock(
            in_dim=hidden_dim * (2**4),
            skip_dim=hidden_dim * (2**3),
            out_dim=hidden_dim * (2**3),
            with_batch_norm=with_batch_norm,
        )
        self.up3 = UpsampleBlock(
            in_dim=hidden_dim * (2**3),
            skip_dim=hidden_dim * (2**2),
            out_dim=hidden_dim * (2**2),
            with_batch_norm=with_batch_norm,
        )
        self.up2 = UpsampleBlock(
            in_dim=hidden_dim * (2**2),
            skip_dim=hidden_dim * (2**1),
            out_dim=hidden_dim * (2**1),
            with_batch_norm=with_batch_norm,
        )
        self.up1 = UpsampleBlock(
            in_dim=hidden_dim * (2**1),
            skip_dim=hidden_dim * (2**0),
            out_dim=hidden_dim * (2**0),
            with_batch_norm=with_batch_norm,
        )

        self.predict = nn.Conv2d(
            hidden_dim, 1, kernel_size=3, stride=1, padding=1, bias=False
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, 0.02 / n)
                # m.weight.data.normal_(0, 0.02)
                kaiming_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, l_img, r_img):
        # l_img: 1 x 3 x 480 x 640
        l_img = (2.0 * (l_img / 255.0) - 1.0).contiguous()
        r_img = (2.0 * (r_img / 255.0) - 1.0).contiguous()

        n, c, h, w = l_img.size()

        align = 2**self.down_factor

        w_pad = (align - (w % align)) % align
        h_pad = (align - (h % align)) % align

        # l_img: 1 x 3 x 480 x 640
        l_img = F.pad(l_img, (0, w_pad, 0, h_pad))
        r_img = F.pad(r_img, (0, w_pad, 0, h_pad))

        # conv1: 1 x C x 240 x 320
        l_conv1 = self.conv1(l_img)
        r_conv1 = self.conv1(r_img)

        # conv2: 1 x 2C x 120 x 160
        l_conv2 = self.conv2(l_conv1)
        r_conv2 = self.conv2(r_conv1)

        # conv_redir: 1 x C x 120 x 160
        l_conv_redir = self.conv_redir(l_conv2)

        # corr_volume: 1 x MAX_DISP x 120 x 160
        corr_volume = make_correlation_volume(
            l_conv2, r_conv2, max_disp=(self.max_disp // 4)
        )

        in_conv3 = torch.concat((l_conv_redir, corr_volume), dim=1)

        # conv3: 1 x 4C x 60 x 80
        out_conv3 = self.conv3(in_conv3)

        # res4: 1 x 8C x 30 x 40
        out_res4 = self.res4(out_conv3)

        # res5: 1 x 16C x 15 x 20
        out_res5 = self.res5(out_res4)

        # res6: 1 x 32C x 7 x 10
        out_res6 = self.res6(out_res5)

        # disp06: 1 x 1 x 7 x 10
        # concat_up5: 1 x 16C x 15 x 20
        disp06, concat_up5 = self.up5(out_res6, out_res5)

        # disp05: 1 x 1 x 15 x 20
        # concat_up4: 1 x 8C x 30 x 40
        disp05, concat_up4 = self.up4(concat_up5, out_res4)

        # disp04: 1 x 1 x 30 x 40
        # concat_up3: 1 x 4C x 60 x 80
        disp04, concat_up3 = self.up3(concat_up4, out_conv3)

        # disp03: 1 x 1 x 60 x 80
        # concat_up2: 1 x 2C x 120 x 160
        disp03, concat_up2 = self.up2(concat_up3, l_conv2)

        # disp02: 1 x 1 x 120 x 160
        # concat_up1: 1 x C x 240 x 320
        disp02, concat_up1 = self.up1(concat_up2, l_conv1)

        # disp01: 1 x 1 x 240 x 320
        # concat_up0: 1 x C x 480 x 640
        disp01 = self.predict(concat_up1)

        multi_scale = [disp06, disp05, disp04, disp03, disp02, disp01]
        multi_scale = [
            -1.0 * disparity_interpolate(disp, l_img.shape[2:])[:, :, :h, :w]
            for disp in multi_scale
        ]
        return multi_scale

    def freeze(self):
        for name, param in self.named_parameters():
            if ("weight" in name) or ("bias" in name):
                param.requires_grad = False

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if "weight" in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if "bias" in name]
