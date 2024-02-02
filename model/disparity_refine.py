import torch
import torch.nn as nn
import torch.nn.functional as F
from model.submodule import *


def disparity_refine_factory(config):
    refine_type = config["type"]
    if refine_type == "dilate":
        return DilateRefineNet(**config["arguments"])
    else:
        raise NotImplementedError(f"invalid refine net type: {refine_type}!")


class BaseRefineNet(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, l_disp, l_fmaps=None, r_fmaps=None):
        raise NotImplementedError


class DilateRefineBlock(nn.Module):

    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        refine_dilates,
        use_warp_feature=True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.refine_dilates = refine_dilates
        self.conv0 = nn.Sequential(
            get_conv2d_3x3(in_dim=self.in_dim,
                           out_dim=self.hidden_dim,
                           stride=1,
                           dilation=1,
                           norm_type="batch",
                           use_relu=True),
            *[
                ResidualBlock(in_dim=self.hidden_dim,
                              out_dim=self.hidden_dim,
                              norm_type="batch",
                              stride=1,
                              dilation=dilation) for dilation in refine_dilates
            ],
            nn.Conv2d(self.hidden_dim, self.out_dim, 3, 1, 1),
        )

        self.use_warp_feature = use_warp_feature
        if self.use_warp_feature:
            self.warp_head = WarpHead()

    def forward(self, l_disp, l_fmap, r_fmap):
        n, _, h, w = l_disp.shape

        if l_fmap.shape[2:] != l_disp.shape[2:]:
            l_fmap = F.interpolate(
                l_fmap,
                (h, w),
                mode="bilinear",
                align_corners=True,
            )

        if self.use_warp_feature:
            if r_fmap.shape[2:] != l_disp.shape[2:]:
                r_fmap = F.interpolate(
                    r_fmap,
                    (h, w),
                    mode="bilinear",
                    align_corners=True,
                )
            r_fmap = self.warp_head(r_fmap, l_disp)

        l_edge = torch.cat((l_disp, l_fmap, r_fmap), dim=1) \
            if self.use_warp_feature else torch.cat((l_disp, l_fmap), dim=1)

        l_edge = self.conv0(l_edge)
        return F.relu(l_disp + l_edge)


class DilateRefineNet(BaseRefineNet):

    def __init__(
        self,
        in_dims,
        hidden_dims,
        use_warp_feature,
        refine_dilates,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.in_dims = in_dims
        self.hidden_dims = hidden_dims
        assert len(in_dims) == len(hidden_dims)
        self.use_warp_feature = use_warp_feature
        self.refine_dilates = refine_dilates

        self.refine_layers = nn.ModuleList([])
        for in_dim, hidden_dim in zip(in_dims, hidden_dims):
            self.refine_layers.append(
                DilateRefineBlock(\
                    in_dim=1 + (2 if use_warp_feature else 1) * in_dim,
                    hidden_dim=hidden_dim,
                    out_dim=1,
                    refine_dilates=refine_dilates,
                    use_warp_feature=use_warp_feature))

    def forward(self, l_disp, l_fmaps=None, r_fmaps=None):
        assert len(l_fmaps) == len(r_fmaps) == len(self.refine_layers)

        l_disp_pyramid = []
        for l_fmap, r_fmap, refine_layer \
            in zip(l_fmaps, r_fmaps, self.refine_layers):
            l_disp = \
                F.interpolate(l_disp, scale_factor=2, mode="bilinear", align_corners=True) * 2.0

            l_disp = refine_layer(l_disp, l_fmap, r_fmap)
            l_disp_pyramid.append(l_disp)
        return l_disp_pyramid
