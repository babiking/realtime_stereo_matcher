import torch
import torch.nn as nn
import torch.nn.functional as F
from model.feature_extract import feature_extract_factory
from model.cost_volume import cost_volume_factory
from model.cost_aggregate import cost_aggregate_factory
from model.disparity_regress import disparity_regress_factory
from model.disparity_refine import disparity_refine_factory


class MobileStereoBase(nn.Module):
    def __init__(self, model_config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.model_config = model_config

        self.model_name = model_config["model_name"]

        self.extract = feature_extract_factory(model_config["feature_extract"])

        self.down_factor = self.extract.get_down_factor()

        self.cost = cost_volume_factory(model_config["cost_volume"])
        self.aggregate = cost_aggregate_factory(model_config["cost_aggregate"])
        self.regress = disparity_regress_factory(model_config["disparity_regress"])
        self.refine = disparity_refine_factory(model_config["disparity_refine"])

    def align(self, img):
        n, c, src_h, src_w = img.shape

        divisor = 2**self.down_factor

        h_pad = (divisor - (src_h % divisor)) % divisor
        w_pad = (divisor - (src_w % divisor)) % divisor
        return h_pad, w_pad

    def pad(self, img):
        h_pad, w_pad = self.align(img)

        if h_pad != 0 or w_pad != 0:
            img = F.pad(img, (0, w_pad, 0, h_pad))
        return img

    def upsample(self, disp, dst_size):
        src_h, src_w = disp.shape[2:]
        dst_h, dst_w = dst_size

        if src_h != dst_h or src_w != dst_w:
            scale = float(dst_w) / src_w
            disp = F.interpolate(
                disp * scale, dst_size, mode="bilinear", align_corners=False
            )
        return disp

    def forward(self, l_img, r_img, is_train=True):
        _, _, src_h, src_w = l_img.shape

        l_img = (2.0 * (l_img / 255.0) - 1.0).contiguous()
        r_img = (2.0 * (r_img / 255.0) - 1.0).contiguous()

        l_img = self.pad(l_img)
        r_img = self.pad(r_img)

        _, _, dst_h, dst_w = l_img.shape

        l_fmaps = self.extract(l_img)
        r_fmaps = self.extract(r_img)

        cost_volume = self.cost(l_fmaps[0], r_fmaps[0])
        cost_volume = self.aggregate(cost_volume)

        l_disp = self.regress(cost_volume, self.cost.max_disp)
        l_disp_pyramid = self.refine(
            l_disp, l_fmaps[1:], r_fmaps[1:], is_train=is_train
        )

        if is_train:
            return l_disp_pyramid
        else:
            if src_h != dst_h or src_w != dst_w:
                l_disp_pyramid[-1] = l_disp_pyramid[-1][:, :, :src_h, :src_w]
            return l_disp_pyramid
