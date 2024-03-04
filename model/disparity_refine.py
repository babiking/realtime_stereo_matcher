import torch
import torch.nn as nn
import torch.nn.functional as F
from model.submodule import *


def disparity_refine_factory(config):
    refine_type = config["type"]
    if refine_type == "dilate":
        return DilateRefineNet(**config["arguments"])
    elif refine_type == "context":
        return ContextRefineNet(**config["arguments"])
    else:
        raise NotImplementedError(f"invalid refine net type: {refine_type}!")


class BaseRefineNet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, l_disp, l_fmaps=None, r_fmaps=None, is_train=True):
        raise NotImplementedError


class ContextRefineNet(BaseRefineNet):
    def __init__(
        self,
        in_dim,
        hidden_dims,
        context_dilates,
        fold_radius,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.context_dilates = context_dilates
        self.fold_radius = fold_radius

        self.context = nn.Sequential(
            get_conv2d_1x1(
                in_dim=self.in_dim,
                out_dim=self.hidden_dims[0],
                stride=1,
                dilation=self.context_dilates[0],
                norm_type="batch",
                use_relu=True,
            ),
            *[
                ResidualBlock(
                    in_dim=self.hidden_dims[i - 1],
                    out_dim=self.hidden_dims[i],
                    norm_type="batch",
                    stride=1,
                    dilation=self.context_dilates[i],
                )
                for i in range(1, len(self.hidden_dims))
            ],
            nn.Conv2d(self.hidden_dims[-1], (self.fold_radius ** 2), 3, 1, 1),
        )

    def upsample(self, l_disp, l_mask):
        """
            Upsample disparity map by convex combination.

            Args:
                [1] l_disp: N x 1 x H x W
                [2] l_mask: N x 9 x (8 * H) x (8 * W)

            Return:
                [1] l_disp: N x 1 x H x W

        l_mask.view([N, 9, 8, H/8, 8, W/8])
        l_mask.softmax(dim=1)

        l_disp.unfold([N, 9, 1, H/8, 1, W/8])

        l_disp = (l_disp * l_mask).sum(dim=1)

        l_disp.reshape([N, 1, H, W])
        """
        n, _, h0, w0 = l_disp.shape
        n, c1, h1, w1 = l_mask.shape

        assert c1 == (self.fold_radius**2)

        assert w1 % w0 == 0
        up_factor = int(w1 / w0)

        l_mask = torch.softmax(l_mask, dim=1)
        l_mask = l_mask.view([n, c1, up_factor, h0, up_factor, w0])

        l_disp_up = F.unfold(
            l_disp, kernel_size=[self.fold_radius, self.fold_radius], padding=1
        ) * float(up_factor)
        l_disp_up = l_disp_up.view([n, c1, 1, h0, 1, w0])
        l_disp_up = (l_disp_up * l_mask).sum(1).squeeze(1)

        l_disp_up = l_disp_up.view([n, 1, h1, w1])
        return l_disp_up

    def forward(self, l_disp, l_fmaps=None, r_fmaps=None, is_train=True):
        assert l_fmaps[-1].shape[-1] % l_disp.shape[-1] == 0
        assert l_fmaps[-1].shape[-2] % l_disp.shape[-2] == 0

        l_mask = self.context(l_fmaps[-1])

        l_disp_up = self.upsample(l_disp, l_mask)
        if is_train:
            return (
                [l_disp, l_disp_up],
                [l_fmaps[0], l_fmaps[-1]],
                [r_fmaps[0], r_fmaps[-1]],
            )
        else:
            return [l_disp_up]


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
            get_conv2d_3x3(
                in_dim=self.in_dim,
                out_dim=self.hidden_dim,
                stride=1,
                dilation=1,
                norm_type="batch",
                use_relu=True,
            ),
            *[
                ResidualBlock(
                    in_dim=self.hidden_dim,
                    out_dim=self.hidden_dim,
                    norm_type="batch",
                    stride=1,
                    dilation=dilation,
                )
                for dilation in refine_dilates
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

        l_edge = (
            torch.cat((l_disp, l_fmap, r_fmap), dim=1)
            if self.use_warp_feature
            else torch.cat((l_disp, l_fmap), dim=1)
        )

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
                DilateRefineBlock(
                    in_dim=1 + (2 if use_warp_feature else 1) * in_dim,
                    hidden_dim=hidden_dim,
                    out_dim=1,
                    refine_dilates=refine_dilates,
                    use_warp_feature=use_warp_feature,
                )
            )

    def forward(self, l_disp, l_fmaps=None, r_fmaps=None, is_train=True):
        assert len(l_fmaps) == len(r_fmaps) == len(self.refine_layers)

        l_disp_pyramid = []
        for i, (l_fmap, r_fmap, refine_layer) in enumerate(
            zip(l_fmaps, r_fmaps, self.refine_layers)
        ):
            l_disp = (
                F.interpolate(
                    l_disp, scale_factor=2, mode="bilinear", align_corners=True
                )
                * 2.0
            )

            l_disp = refine_layer(l_disp, l_fmap, r_fmap)

            if (not is_train) and i != len(self.refine_layers) - 1:
                continue

            l_disp_pyramid.append(l_disp)

        if is_train:
            return [l_disp] + l_disp_pyramid, l_fmaps, r_fmaps
        else:
            return l_disp_pyramid
