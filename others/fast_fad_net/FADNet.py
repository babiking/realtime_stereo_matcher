from __future__ import print_function
import torch
import torch.nn as nn
from others.fast_fad_net.DispNetC import ExtractNet, CUNet
from others.fast_fad_net.DispNetRes import DispNetRes
from others.fast_fad_net.submodules import *


class FADNet(nn.Module):
    def __init__(
        self,
        resBlock=True,
        maxdisp=192,
        input_channel=3,
        encoder_ratio=16,
        decoder_ratio=16,
        use_dispnetc_only=False,
    ):
        super(FADNet, self).__init__()
        self.input_channel = input_channel
        self.maxdisp = maxdisp
        self.resBlock = resBlock
        self.eratio = encoder_ratio
        self.dratio = decoder_ratio
        self.use_dispnetc_only = use_dispnetc_only

        # First Block (Extract)
        self.extract_network = ExtractNet(
            resBlock=resBlock,
            maxdisp=self.maxdisp,
            input_channel=input_channel,
            encoder_ratio=encoder_ratio,
            decoder_ratio=decoder_ratio,
        )

        # Second Block (CUNet)
        self.cunet = CUNet(
            resBlock=resBlock,
            maxdisp=self.maxdisp,
            input_channel=input_channel,
            encoder_ratio=encoder_ratio,
            decoder_ratio=decoder_ratio,
        )

        # Third Block (DispNetRes), input is 11 channels(img0, img1, img1->img0, flow, diff-mag)
        if not self.use_dispnetc_only:
            in_planes = 3 * 3 + 1 + 1
            self.dispnetres = DispNetRes(
                in_planes,
                resBlock=resBlock,
                input_channel=input_channel,
                encoder_ratio=encoder_ratio,
                decoder_ratio=decoder_ratio,
            )

        self.relu = nn.ReLU(inplace=False)

    def load_data_item(self, img):
        n, c, h, w = img.shape

        img = torch.reshape(img, shape=(n, c, h * w))
        img = (img - torch.mean(img, dim=2, keepdim=True)) / torch.std(
            img, dim=2, keepdim=True
        )

        img = torch.reshape(img, shape=(n, c, h, w))
        return img

    def forward(self, img_left, img_right, is_train=False):
        inputs = torch.concat((img_left, img_right), dim=1)

        # extract features
        conv1_l, conv2_l, conv3a_l, conv3a_r = self.extract_network(inputs)

        # build corr
        out_corr = build_corr(conv3a_l, conv3a_r, max_disp=self.maxdisp // 8 + 16)
        # generate first-stage flows
        dispnetc_flows = self.cunet(inputs, conv1_l, conv2_l, conv3a_l, out_corr)
        dispnetc_final_flow = dispnetc_flows[0]

        if not self.use_dispnetc_only:
            # warp img1 to img0; magnitude of diff between img0 and warped_img1,
            resampled_img1 = warp_right_to_left(
                inputs[:, self.input_channel :, :, :], -dispnetc_final_flow
            )
            diff_img0 = inputs[:, : self.input_channel, :, :] - resampled_img1
            norm_diff_img0 = channel_length(diff_img0)

            # concat img0, img1, img1->img0, flow, diff-img
            inputs_net2 = torch.cat(
                (inputs, resampled_img1, dispnetc_final_flow, norm_diff_img0), dim=1
            )

            # dispnetres
            # dispnetres_flows = self.dispnetres(inputs_net2, dispnetc_flows)
            dispnetres_flows = self.dispnetres(inputs_net2, dispnetc_flows)

            index = 0
            dispnetres_final_flow = dispnetres_flows[index]

        if self.training and is_train:
            l_disps = (
                [
                    dispnetc_flows[len(dispnetc_flows) - 1 - i]
                    for i in range(len(dispnetc_flows))
                ]
                if self.use_dispnetc_only
                else [
                    dispnetres_flows[len(dispnetres_flows) - 1 - i]
                    for i in range(len(dispnetres_flows))
                ]
            )
            return l_disps, [None] * len(l_disps), [None] * len(l_disps)
        else:
            return (
                [dispnetc_final_flow]
                if self.use_dispnetc_only
                else [dispnetres_final_flow]
            )

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if "weight" in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if "bias" in name]
