from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Function
from torch.nn import init
from torch.nn.init import kaiming_normal

# from correlation_package.modules.corr import Correlation1d # from PWC-Net
from others.mobile_disp_net_c.submodules import *


class CostVolume2D(nn.Module):
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
        cost_item = torch.mean(l_fmap * r_fmap, dim=1, keepdim=True)
        return cost_item

    def make_cost_volume_naive(self, l_fmap, r_fmap):
        # left: 1 x 32 x 60 x 80
        # right: 1 x 32 x 60 x 80
        # max_disp: 24
        # cost_volume: 1 x 32 x 24 x 60 x 80
        n, c, h, w = l_fmap.shape

        cost_volume = torch.zeros(
            size=[n, self.max_disp, h, w],
            dtype=l_fmap.dtype,
            device=l_fmap.device,
        )

        # for any disparity d:
        #   cost_volume[:, :, d, :, :d] = 1.0
        #   cost_volume[:, :, d, :, d:] = left[:, :, :, d:] - right[:, :, :, :-d]
        cost_volume[:, 0, :, :] = self.get_cost_item(l_fmap, r_fmap).squeeze(1)
        for d in range(1, self.max_disp):
            cost_volume[:, d, :, d:] = self.get_cost_item(
                l_fmap[:, :, :, d:], r_fmap[:, :, :, :-d]
            ).squeeze(1)

        # cost_volume: 1 x 32 x 24 x 60 x 80
        return cost_volume

    def make_cost_volume_conv2d(self, l_fmap, r_fmap):
        cost_volume = []
        for d in range(self.max_disp):
            if d == 0:
                cost_volume.append(self.get_cost_item(l_fmap, r_fmap))
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

                cost_volume.append(F.pad(cost_item, [d, 0, 0, 0], value=0.0))
        cost_volume = torch.concat(cost_volume, dim=1)
        return cost_volume

    def forward(self, l_fmap, r_fmap, use_naive):
        return (
            self.make_cost_volume_naive(l_fmap, r_fmap)
            if use_naive
            else self.make_cost_volume_conv2d(l_fmap, r_fmap)
        )


class MobileDispNetC(nn.Module):
    def __init__(
        self,
        batchNorm=True,
        lastRelu=True,
        resBlock=True,
        maxdisp=192,
        input_channel=3,
        get_features=False,
    ):
        super(MobileDispNetC, self).__init__()

        self.batchNorm = batchNorm
        self.lastRelu = lastRelu
        self.input_channel = input_channel
        self.maxdisp = maxdisp
        self.get_features = get_features
        self.relu = nn.ReLU(inplace=False)

        # shrink and extract features
        self.conv1 = conv(self.input_channel, 64, 7, 2)
        if resBlock:
            self.conv2 = ResBlock(64, 128, stride=2)
            self.conv3 = ResBlock(128, 256, stride=2)
            self.conv_redir = ResBlock(256, 32, stride=1)
        else:
            self.conv2 = conv(64, 128, stride=2)
            self.conv3 = conv(128, 256, stride=2)
            self.conv_redir = conv(256, 32, stride=1)
        self.corr_activation = nn.LeakyReLU(0.1, inplace=True)
        if resBlock:
            self.conv3_1 = ResBlock(72, 128)
            self.conv4 = ResBlock(128, 256, stride=2)
            self.conv5 = ResBlock(256, 512, stride=2)
            self.conv6 = ResBlock(512, 512, stride=2)
        else:
            self.conv3_1 = conv(72, 128)
            self.conv4 = conv(128, 256, stride=2)
            self.conv5 = conv(256, 512, stride=2)
            self.conv6 = conv(512, 512, stride=2)

        self.pred_flow6 = predict_flow(512)

        # iconv with deconv
        self.iconv5 = nn.ConvTranspose2d(512 + 256 + 1, 256, 3, 1, 1)
        self.iconv4 = nn.ConvTranspose2d(256 + 128 + 1, 128, 3, 1, 1)
        self.iconv3 = nn.ConvTranspose2d(128 + 64 + 1, 64, 3, 1, 1)
        self.iconv2 = nn.ConvTranspose2d(128 + 32 + 1, 32, 3, 1, 1)
        self.iconv1 = nn.ConvTranspose2d(64 + 16 + 1, 16, 3, 1, 1)
        self.iconv0 = nn.ConvTranspose2d(17 + self.input_channel, 16, 3, 1, 1)

        # expand and produce disparity
        self.upconv5 = deconv(512, 256)
        self.upflow6to5 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_flow5 = predict_flow(256)

        self.upconv4 = deconv(256, 128)
        self.upflow5to4 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_flow4 = predict_flow(128)

        self.upconv3 = deconv(128, 64)
        self.upflow4to3 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_flow3 = predict_flow(64)

        self.upconv2 = deconv(64, 32)
        self.upflow3to2 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_flow2 = predict_flow(32)

        self.upconv1 = deconv(32, 16)
        self.upflow2to1 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_flow1 = predict_flow(16)

        self.upconv0 = deconv(16, 16)
        self.upflow1to0 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        if self.maxdisp == -1:
            self.pred_flow0 = predict_flow(16)
        else:
            self.disp_expand = ResBlock(16, self.maxdisp)

        self.cost_volume = CostVolume2D(hidden_dim=256, max_disp=40)

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

        # self.freeze()

    def forward(self, img_left, img_right, is_train=True):
        conv1_l = self.conv1(img_left)  # 3-64
        conv2_l = self.conv2(conv1_l)  # 64-128
        conv3a_l = self.conv3(conv2_l)  # 128-256

        conv1_r = self.conv1(img_right)
        conv2_r = self.conv2(conv1_r)
        conv3a_r = self.conv3(conv2_r)

        # Correlate corr3a_l and corr3a_r
        # out_corr = self.corr(conv3a_l, conv3a_r)
        out_corr = self.cost_volume(conv3a_l, conv3a_r, is_train)
        out_corr = self.corr_activation(out_corr)
        out_conv3a_redir = self.conv_redir(conv3a_l)  # 256-32
        in_conv3b = torch.cat((out_conv3a_redir, out_corr), 1)  # 32+40

        conv3b = self.conv3_1(in_conv3b)  # 72-128
        conv4a = self.conv4(conv3b)  # 128-256
        conv5a = self.conv5(conv4a)  # 256-512
        conv6a = self.conv6(conv5a)  # 512-512

        pr6 = self.pred_flow6(conv6a)  # 512-1
        upconv5 = self.upconv5(conv6a)  # 512-256
        upflow6 = self.upflow6to5(pr6)  # 1-1
        concat5 = torch.cat((upconv5, upflow6, conv5a), 1)
        iconv5 = self.iconv5(concat5)  # 256+1+512-256

        pr5 = self.pred_flow5(iconv5)  # 256-1
        upconv4 = self.upconv4(iconv5)  # 256-128
        upflow5 = self.upflow5to4(pr5)  # 1-1
        concat4 = torch.cat((upconv4, upflow5, conv4a), 1)
        iconv4 = self.iconv4(concat4)  # 128+1+256-128

        pr4 = self.pred_flow4(iconv4)  # 128-1
        upconv3 = self.upconv3(iconv4)  # 128-64
        upflow4 = self.upflow4to3(pr4)  # 1-1
        concat3 = torch.cat((upconv3, upflow4, conv3b), 1)
        iconv3 = self.iconv3(concat3)  # 64+1+128-64

        pr3 = self.pred_flow3(iconv3)  # 64-1
        upconv2 = self.upconv2(iconv3)  # 64-32
        upflow3 = self.upflow3to2(pr3)  # 1-1
        concat2 = torch.cat((upconv2, upflow3, conv2_l), 1)
        iconv2 = self.iconv2(concat2)  # 32+1+128, 32

        pr2 = self.pred_flow2(iconv2)  # 32-1
        upconv1 = self.upconv1(iconv2)  # 32-16
        upflow2 = self.upflow2to1(pr2)  # 1-1
        concat1 = torch.cat((upconv1, upflow2, conv1_l), 1)
        iconv1 = self.iconv1(concat1)  # 16+1+64-16

        pr1 = self.pred_flow1(iconv1)  # 16-1
        upconv0 = self.upconv0(iconv1)  # 16-16
        upflow1 = self.upflow1to0(pr1)  # 1-1
        concat0 = torch.cat((upconv0, upflow1, img_left), 1)
        iconv0 = self.iconv0(concat0)  # 16+1+3-16

        # predict flow
        if self.maxdisp == -1:
            pr0 = self.pred_flow0(iconv0)
            pr0 = self.relu(pr0)
        else:
            pr0 = self.disp_expand(iconv0)
            pr0 = F.softmax(pr0, dim=1)
            pr0 = disparity_regression(pr0, self.maxdisp)

        if is_train:
            disps = [pr5, pr4, pr3, pr2, pr1, pr0]
            return disps, [None] * len(disps), [None] * len(disps)
        else:
            return [pr0]

    def freeze(self):
        for name, param in self.named_parameters():
            if ("weight" in name) or ("bias" in name):
                param.requires_grad = False

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if "weight" in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if "bias" in name]