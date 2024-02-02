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
        self.regress = \
            disparity_regress_factory(model_config["disparity_regress"])
        self.refine = \
            disparity_refine_factory(model_config["disparity_refine"])

    def align(self, img, down_factor):
        n, c, src_h, src_w = img.shape

        divisor = 2**down_factor

        h_pad = (divisor - (src_h % divisor)) % divisor
        w_pad = (divisor - (src_w % divisor)) % divisor
        return h_pad, w_pad

    def upsample(self, disp, dst_size):
        src_h, src_w = disp.shape[2:]
        dst_h, dst_w = dst_size

        scale = float(dst_w) / src_w
        disp = F.interpolate(disp * scale,
                             dst_size,
                             mode="bilinear",
                             align_corners=True)
        return disp

    def forward(self, l_img, r_img):
        _, _, h, w = l_img.shape

        l_img = (2.0 * (l_img / 255.0) - 1.0).contiguous()
        r_img = (2.0 * (r_img / 255.0) - 1.0).contiguous()

        h_pad, w_pad = self.align(l_img, self.down_factor)
        l_img = F.pad(l_img, (0, w_pad, 0, h_pad))
        r_img = F.pad(r_img, (0, w_pad, 0, h_pad))

        l_fmaps = self.extract(l_img)
        r_fmaps = self.extract(r_img)

        cost_volume = self.cost(l_fmaps[0], r_fmaps[0])
        cost_volume = self.aggregate(cost_volume)

        l_disp = self.regress(cost_volume)
        l_disp_pyramid = self.refine(l_disp, l_fmaps[1:], r_fmaps[1:])
        l_disp_pyramid = [
            -1.0 * self.upsample(l_disp, l_img.shape[2:])[:, :, :h, :w]
            for l_disp in l_disp_pyramid
        ]
        return l_disp_pyramid
