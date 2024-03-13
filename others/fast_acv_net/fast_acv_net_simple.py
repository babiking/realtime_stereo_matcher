from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from others.fast_acv_net.submodule import *
import math
import gc
import time
import timm


class SubModule(nn.Module):
    def __init__(self):
        super(SubModule, self).__init__()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.Conv3d):
                n = (
                    m.kernel_size[0]
                    * m.kernel_size[1]
                    * m.kernel_size[2]
                    * m.out_channels
                )
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Feature(SubModule):
    def __init__(self):
        super(Feature, self).__init__()
        pretrained = True
        model = timm.create_model(
            "mobilenetv2_100", pretrained=pretrained, features_only=True
        )
        layers = [1, 2, 3, 5, 6]
        chans = [16, 24, 32, 96, 160]
        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1
        self.act1 = nn.ReLU()  # model.act1

        self.block0 = torch.nn.Sequential(*model.blocks[0 : layers[0]])
        self.block1 = torch.nn.Sequential(*model.blocks[layers[0] : layers[1]])
        self.block2 = torch.nn.Sequential(*model.blocks[layers[1] : layers[2]])
        self.block3 = torch.nn.Sequential(*model.blocks[layers[2] : layers[3]])
        self.block4 = torch.nn.Sequential(*model.blocks[layers[3] : layers[4]])

    def forward(self, x):
        x = self.act1(self.bn1(self.conv_stem(x)))
        x2 = self.block0(x)
        x4 = self.block1(x2)
        x8 = self.block2(x4)
        x16 = self.block3(x8)
        x32 = self.block4(x16)
        return [x4, x8, x16, x32]


class FeatUp(SubModule):
    def __init__(self):
        super(FeatUp, self).__init__()
        chans = [16, 24, 32, 96, 160]
        self.deconv32_16 = Conv2x(chans[4], chans[3], deconv=True, concat=True)
        self.deconv16_8 = Conv2x(chans[3] * 2, chans[2], deconv=True, concat=True)
        self.deconv8_4 = Conv2x(chans[2] * 2, chans[1], deconv=True, concat=True)
        self.conv4 = BasicConv(
            chans[1] * 2, chans[1] * 2, kernel_size=3, stride=1, padding=1
        )

        self.weight_init()

    def forward(self, featL, featR=None):
        x4, x8, x16, x32 = featL
        y4, y8, y16, y32 = featR

        x16 = self.deconv32_16(x32, x16)
        y16 = self.deconv32_16(y32, y16)

        x8 = self.deconv16_8(x16, x8)
        y8 = self.deconv16_8(y16, y8)

        x4 = self.deconv8_4(x8, x4)
        y4 = self.deconv8_4(y8, y4)

        x4 = self.conv4(x4)
        y4 = self.conv4(y4)

        return [x4, x8, x16, x32], [y4, y8, y16, y32]


class channelAtt(SubModule):
    def __init__(self, cv_chan, im_chan):
        super(channelAtt, self).__init__()

        self.im_att = nn.Sequential(
            BasicConv(im_chan, im_chan // 2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(im_chan // 2, cv_chan, 1),
        )

        self.weight_init()

    def forward(self, cv, im):
        channel_att = self.im_att(im).unsqueeze(2)
        cv = torch.sigmoid(channel_att) * cv
        return cv


class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(
            BasicConv(
                in_channels,
                in_channels * 2,
                is_3d=True,
                bn=True,
                relu=True,
                kernel_size=3,
                padding=1,
                stride=2,
                dilation=1,
            ),
            BasicConv(
                in_channels * 2,
                in_channels * 2,
                is_3d=True,
                bn=True,
                relu=True,
                kernel_size=3,
                padding=1,
                stride=1,
                dilation=1,
            ),
        )

        self.conv2 = nn.Sequential(
            BasicConv(
                in_channels * 2,
                in_channels * 4,
                is_3d=True,
                bn=True,
                relu=True,
                kernel_size=3,
                padding=1,
                stride=2,
                dilation=1,
            ),
            BasicConv(
                in_channels * 4,
                in_channels * 4,
                is_3d=True,
                bn=True,
                relu=True,
                kernel_size=3,
                padding=1,
                stride=1,
                dilation=1,
            ),
        )

        self.conv2_up = BasicConv(
            in_channels * 4,
            in_channels * 2,
            deconv=True,
            is_3d=True,
            bn=True,
            relu=True,
            kernel_size=(4, 4, 4),
            padding=(1, 1, 1),
            stride=(2, 2, 2),
        )

        self.conv1_up = BasicConv(
            in_channels * 2,
            1,
            deconv=True,
            is_3d=True,
            bn=False,
            relu=False,
            kernel_size=(4, 4, 4),
            padding=(1, 1, 1),
            stride=(2, 2, 2),
        )

        self.agg = nn.Sequential(
            BasicConv(
                in_channels * 4,
                in_channels * 2,
                is_3d=True,
                kernel_size=1,
                padding=0,
                stride=1,
            ),
            BasicConv(
                in_channels * 2,
                in_channels * 2,
                is_3d=True,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            BasicConv(
                in_channels * 2,
                in_channels * 2,
                is_3d=True,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
        )

        self.feature_att_8 = channelAtt(in_channels * 2, 96)
        self.feature_att_16 = channelAtt(in_channels * 4, 192)
        self.feature_att_up_8 = channelAtt(in_channels * 2, 96)

    def forward(self, x, imgs):
        conv1 = self.conv1(x)
        conv1 = self.feature_att_8(conv1, imgs[1])

        conv2 = self.conv2(conv1)
        conv2 = self.feature_att_16(conv2, imgs[2])

        conv2_up = self.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg(conv1)
        conv1 = self.feature_att_up_8(conv1, imgs[1])

        conv = self.conv1_up(conv1)

        return conv


class hourglass_att(nn.Module):
    def __init__(self, in_channels):
        super(hourglass_att, self).__init__()

        self.conv1 = nn.Sequential(
            BasicConv(
                in_channels,
                in_channels * 2,
                is_3d=True,
                bn=True,
                relu=True,
                kernel_size=3,
                padding=1,
                stride=2,
                dilation=1,
            ),
            BasicConv(
                in_channels * 2,
                in_channels * 2,
                is_3d=True,
                bn=True,
                relu=True,
                kernel_size=3,
                padding=1,
                stride=1,
                dilation=1,
            ),
        )

        self.conv2 = nn.Sequential(
            BasicConv(
                in_channels * 2,
                in_channels * 4,
                is_3d=True,
                bn=True,
                relu=True,
                kernel_size=3,
                padding=1,
                stride=2,
                dilation=1,
            ),
            BasicConv(
                in_channels * 4,
                in_channels * 4,
                is_3d=True,
                bn=True,
                relu=True,
                kernel_size=3,
                padding=1,
                stride=1,
                dilation=1,
            ),
        )

        self.conv2_up = BasicConv(
            in_channels * 4,
            in_channels * 2,
            deconv=True,
            is_3d=True,
            bn=True,
            relu=True,
            kernel_size=(4, 4, 4),
            padding=(1, 1, 1),
            stride=(2, 2, 2),
        )

        self.conv1_up_ = BasicConv(
            in_channels * 2,
            in_channels,
            deconv=True,
            is_3d=True,
            bn=True,
            relu=True,
            kernel_size=(4, 4, 4),
            padding=(1, 1, 1),
            stride=(2, 2, 2),
        )
        self.conv_final = nn.Conv3d(in_channels, 1, 3, 1, 1, bias=False)

        self.agg = nn.Sequential(
            BasicConv(
                in_channels * 4,
                in_channels * 2,
                is_3d=True,
                kernel_size=1,
                padding=0,
                stride=1,
            ),
            BasicConv(
                in_channels * 2,
                in_channels * 2,
                is_3d=True,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
            BasicConv(
                in_channels * 2,
                in_channels * 2,
                is_3d=True,
                kernel_size=3,
                padding=1,
                stride=1,
            ),
        )

        self.feature_att_16 = channelAtt(in_channels * 2, 192)
        self.feature_att_32 = channelAtt(in_channels * 4, 160)
        self.feature_att_up_16 = channelAtt(in_channels * 2, 192)

    def forward(self, x, imgs):
        conv1 = self.conv1(x)
        conv1 = self.feature_att_16(conv1, imgs[2])

        conv2 = self.conv2(conv1)
        conv2 = self.feature_att_32(conv2, imgs[3])

        conv2_up = self.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg(conv1)
        conv1 = self.feature_att_up_16(conv1, imgs[2])

        conv = self.conv1_up_(conv1)
        conv = self.conv_final(conv)

        return conv


class GroupwiseCostVolume3D(nn.Module):
    def __init__(self, hidden_dim, max_disp, num_cost_groups, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.hidden_dim = hidden_dim
        self.max_disp = max_disp
        self.num_cost_groups = num_cost_groups

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
        n, c, h, w = l_fmap.shape
        assert c % self.num_cost_groups == 0
        ch_per_group = c // self.num_cost_groups
        l_fmap = l_fmap.view([n, self.num_cost_groups, ch_per_group, h, w])
        r_fmap = r_fmap.view([n, self.num_cost_groups, ch_per_group, h, w])
        cost_item = (
            l_fmap
            / (torch.norm(l_fmap, p=2, dim=2).unsqueeze(2) + 1e-05)
            * r_fmap
            / (torch.norm(r_fmap, p=2, dim=2).unsqueeze(2) + 1e-05)
        ).mean(dim=2)
        return cost_item

    def make_cost_volume_naive(self, l_fmap, r_fmap):
        # left: 1 x 32 x 60 x 80
        # right: 1 x 32 x 60 x 80
        # max_disp: 24
        # cost_volume: 1 x 32 x 24 x 60 x 80
        n, c, h, w = l_fmap.shape

        cost_volume = torch.zeros(
            size=[n, self.num_cost_groups, self.max_disp, h, w],
            dtype=l_fmap.dtype,
            device=l_fmap.device,
        )

        # for any disparity d:
        #   cost_volume[:, :, d, :, :d] = 1.0
        #   cost_volume[:, :, d, :, d:] = left[:, :, :, d:] - right[:, :, :, :-d]
        cost_volume[:, :, 0, :, :] = self.get_cost_item(l_fmap, r_fmap)
        for d in range(1, self.max_disp):
            cost_volume[:, :, d, :, d:] = self.get_cost_item(
                l_fmap[:, :, :, d:], r_fmap[:, :, :, :-d]
            )

        # cost_volume: 1 x 32 x 24 x 60 x 80
        return cost_volume

    def make_cost_volume_conv2d(self, l_fmap, r_fmap):
        cost_volume = []
        for d in range(self.max_disp):
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
                    F.pad(cost_item, [d, 0, 0, 0], value=0.0).unsqueeze(2)
                )
        cost_volume = torch.concat(cost_volume, dim=2)
        return cost_volume

    def forward(self, l_fmap, r_fmap, use_naive):
        return (
            self.make_cost_volume_naive(l_fmap, r_fmap)
            if use_naive
            else self.make_cost_volume_conv2d(l_fmap, r_fmap)
        )


class FastACVNetSimple(nn.Module):
    def __init__(self, max_disp, use_concat_volume, use_topk_sort, use_warp_score):
        super(FastACVNetSimple, self).__init__()
        self.max_disp = max_disp
        self.use_concat_volume = use_concat_volume
        self.use_topk_sort = use_topk_sort
        self.use_warp_score = use_warp_score
        self.feature = Feature()
        self.feature_up = FeatUp()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(2 * torch.ones(1))

        self.patch = nn.Conv3d(
            12,
            12,
            kernel_size=(1, 3, 3),
            stride=1,
            dilation=1,
            groups=12,
            padding=(0, 1, 1),
            bias=False,
        )

        self.stem_2 = nn.Sequential(
            BasicConv(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.stem_4 = nn.Sequential(
            BasicConv(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(),
        )

        self.stem_8 = nn.Sequential(
            BasicConv(48, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.spx = nn.Sequential(
            nn.ConvTranspose2d(2 * 32, 9, kernel_size=4, stride=2, padding=1),
        )
        self.spx_2 = Conv2x(24, 32, True)
        self.spx_4 = nn.Sequential(
            BasicConv(96, 24, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(24, 24, 3, 1, 1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(),
        )

        self.corr_feature_att_8 = channelAtt(12, 96)

        if self.use_concat_volume:
            self.concat_feature = nn.Sequential(
                BasicConv(96, 32, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(32, 16, 3, 1, 1, bias=False),
            )
            self.concat_feature_att_4 = channelAtt(16, 96)
            self.hourglass = hourglass(16)
            self.concat_stem = BasicConv(
                32, 16, is_3d=True, kernel_size=3, stride=1, padding=1
            )

        self.hourglass_att = hourglass_att(12)
        self.propagation = Propagation()
        self.propagation_prob = Propagation_prob()

        self.cost_builder = GroupwiseCostVolume3D(
            hidden_dim=96, max_disp=max_disp // 8, num_cost_groups=12
        )

    def concat_volume_generator(self, left_input, right_input, disparity_samples):
        right_feature_map, left_feature_map = SpatialTransformer_grid(
            left_input, right_input, disparity_samples
        )
        concat_volume = torch.cat((left_feature_map, right_feature_map), dim=1)
        return concat_volume

    def forward(self, left, right, is_train=True):
        features_left = self.feature(left)
        features_right = self.feature(right)
        features_left, features_right = self.feature_up(features_left, features_right)

        stem_2x = self.stem_2(left)
        stem_4x = self.stem_4(stem_2x)
        stem_8x = self.stem_8(stem_4x)

        stem_2y = self.stem_2(right)
        stem_4y = self.stem_4(stem_2y)
        stem_8y = self.stem_8(stem_4y)

        features_left[0] = torch.cat((features_left[0], stem_4x), 1)
        features_right[0] = torch.cat((features_right[0], stem_4y), 1)

        features_left[1] = torch.cat((features_left[1], stem_8x), 1)
        features_right[1] = torch.cat((features_right[1], stem_8y), 1)

        # spx_pred: used from context upsample 4x
        xspx = self.spx_4(features_left[0])
        xspx = self.spx_2(xspx, stem_2x)
        spx_pred = self.spx(xspx)
        spx_pred = F.softmax(spx_pred, 1)

        # cost_volume_8x: 1 x 12 x 24 x 64 x 80, 8x
        cost_volume_8x = self.cost_builder(
            features_left[1], features_right[1], use_naive=is_train
        )
        cost_volume_8x = self.patch(cost_volume_8x)

        # cost_volume_8x: 1 x 12 x 24 x 64 x 80, 8x, left image feature guided attention weights
        cost_volume_8x = self.corr_feature_att_8(cost_volume_8x, features_left[1])
        # cost_weights_8x: 1 x 1 x 24 x 64 x 80, 8x, left image features guided hourglass filtering
        cost_weights_8x = self.hourglass_att(cost_volume_8x, features_left)

        # cost_weights_4x: 1 x 1 x 48 x 120 x 240, 4x, cost volume, V_init
        cost_weights_4x = F.interpolate(
            cost_weights_8x,
            [self.max_disp // 4, left.shape[2] // 4, left.shape[3] // 4],
            mode="trilinear",
        )

        # disp_probs_4x: 1 x 48 x 120 x 160, D_init
        disp_probs_4x = F.softmax(cost_weights_4x.squeeze(1), dim=1)
        # disp_init_4x: 1 x 1 x 120 x 160, D_init
        disp_init_4x = disparity_regression(disp_probs_4x, self.max_disp // 4)
        # disp_var_4x: 1 x 1 x 120 x 160, U_i, confidence uncertainty,
        disp_var_4x = disparity_variance(
            disp_probs_4x, self.max_disp // 4, disp_init_4x
        )
        disp_var_4x = torch.sigmoid(self.beta + self.gamma * disp_var_4x)
        # disp_var_4x_m: 1 x 5 x 120 x 160
        disp_var_4x = self.propagation(disp_var_4x)

        if self.use_warp_score:
            # disp_init_4x_m: 1 x 5 x 120 x 160
            disp_init_4x = self.propagation(disp_init_4x)
            # left_feature_x4: 1 x 96 x 5 x 120 x 160
            right_feature_x4, left_feature_x4 = SpatialTransformer_grid(
                stem_4x, stem_4y, disp_init_4x
            )
            # disp_match_4x_m: 1 x 5 x 120 x 160
            disp_match_4x = (left_feature_x4 * right_feature_x4).mean(dim=1)
        else:
            disp_match_4x = 1.0

        # disp_match_4x: 1 x 5 x 120 x 160, cross shape propagation weights, W_m_i
        disp_match_4x = torch.softmax(disp_match_4x * disp_var_4x, dim=1)

        # cost_weights_4x: 1 x 5 x 48 x 120 x 160, cost volume after VAP, V_p_i_d
        cost_weights_4x = self.propagation_prob(cost_weights_4x)
        cost_weights_4x = cost_weights_4x * disp_match_4x.unsqueeze(2)
        # cost_weights_4x: 1 x 48 x 120 x 160, cost volume after VAP, V_p_i_d
        cost_weights_4x = torch.sum(cost_weights_4x, dim=1)

        # disp_probs_4x: 1 x 48 x 120 x 160, disparity probability
        disp_probs_4x = F.softmax(cost_weights_4x, dim=1)

        if self.use_topk_sort:
            _, ind = disp_probs_4x.sort(1, True)
            k = self.max_disp // 4 // 2
            ind_k = ind[:, :k]
            ind_k = ind_k.sort(1, False)[0]
            # disp_probs_topk_4x: 1 x 24 x 120 x 160, disparity top-k probability
            disp_probs_topk_4x = torch.gather(disp_probs_4x, 1, ind_k)
            # disp_vals_topk_4x: 1 x 24 x 120 x 160, disparity top-k value
            disp_vals_topk_4x = ind_k.float()

        if self.use_concat_volume:
            if not self.use_topk_sort:
                disp_probs_topk_4x = disp_probs_4x
                disp_vals_topk_4x = torch.arange(
                    0,
                    self.max_disp // 4,
                    dtype=disp_probs_4x.dtype,
                    device=disp_probs_4x.device,
                )
                n, _, h, w = left.shape
                disp_vals_topk_4x = disp_vals_topk_4x.view(
                    [1, self.max_disp // 4, 1, 1]
                ).repeat(n, 1, h // 4, w // 4)

            concat_features_left = self.concat_feature(features_left[0])
            concat_features_right = self.concat_feature(features_right[0])
            # concat_volume: 1 x 32 x 24 x 120 x 160, concat volume
            concat_volume = self.concat_volume_generator(
                concat_features_left, concat_features_right, disp_vals_topk_4x
            )
            concat_volume = disp_probs_topk_4x * concat_volume
            concat_volume = self.concat_stem(concat_volume)
            concat_volume = self.concat_feature_att_4(concat_volume, features_left[0])
            seman_weights_4x = self.hourglass(concat_volume, features_left).squeeze(1)

        if self.use_topk_sort:
            cost_weights_4x = torch.gather(cost_weights_4x, 1, ind_k)
            disp_probs_topk_4x = F.softmax(cost_weights_4x, dim=1)
            l_disp_cost_4x = torch.sum(
                disp_probs_topk_4x * disp_vals_topk_4x, dim=1, keepdim=True
            )
        else:
            l_disp_cost_4x = disparity_regression(
                disp_probs_4x, maxdisp=self.max_disp // 4
            )
        l_disp_cost_up = context_upsample(l_disp_cost_4x, spx_pred) * 4.0

        if self.use_concat_volume:
            if self.use_topk_sort:
                l_disp_seman_4x = regression_topk(
                    seman_weights_4x, disp_vals_topk_4x, 2
                )
            else:
                l_disp_seman_4x = disparity_regression(
                    torch.softmax(seman_weights_4x, dim=1), maxdisp=self.max_disp // 4
                )
            l_disp_seman_up = context_upsample(l_disp_seman_4x, spx_pred) * 4.0

        if self.training:
            if self.use_concat_volume:
                return (
                    [l_disp_cost_4x, l_disp_seman_4x, l_disp_cost_up, l_disp_seman_up],
                    [None] * 4,
                    [None] * 4,
                )
            else:
                return (
                    [l_disp_cost_4x, l_disp_cost_up],
                    [None] * 4,
                    [None] * 4,
                )
        else:
            return [l_disp_seman_up] if self.use_concat_volume else [l_disp_cost_up]
