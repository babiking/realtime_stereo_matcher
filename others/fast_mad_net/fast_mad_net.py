import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from others.fast_mad_net.submodules import BasicConv


class FeatureExtract(nn.Module):
    def __init__(
        self,
        in_dim=3,
        hidden_dims=[16, 32, 64, 96, 128, 192],
        unify_dim=64,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.unify_dim = unify_dim

        self.down_layers = nn.ModuleList([])
        for i in range(len(hidden_dims)):
            self.down_layers.append(
                nn.Sequential(
                    BasicConv(
                        in_channels=in_dim if i == 0 else hidden_dims[i - 1],
                        out_channels=hidden_dims[i],
                        deconv=False,
                        is_3d=False,
                        bn=True,
                        relu=True,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    BasicConv(
                        in_channels=hidden_dims[i],
                        out_channels=hidden_dims[i],
                        deconv=False,
                        is_3d=False,
                        bn=True,
                        relu=True,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                )
            )

        self.unify_layers = nn.ModuleList([])
        for layer_dim in [in_dim] + hidden_dims:
            self.unify_layers.append(
                nn.Sequential(
                    BasicConv(
                        in_channels=layer_dim,
                        out_channels=unify_dim,
                        deconv=False,
                        is_3d=False,
                        bn=True,
                        relu=True,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    BasicConv(
                        in_channels=unify_dim,
                        out_channels=unify_dim,
                        deconv=False,
                        is_3d=False,
                        bn=True,
                        relu=True,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    ),
                )
            )

    def forward(self, fmap):
        fmap_pyramid = [self.unify_layers[0](fmap)]

        for i, down_layer in enumerate(self.down_layers):
            fmap = down_layer(fmap)

            fmap_pyramid.append(self.unify_layers[i + 1](fmap))
        return fmap_pyramid


class Warp1DOp(nn.Module):
    def __init__(
        self, mode="bilinear", padding_mode="zeros", align_corners=True, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners

    def get_random_inputs(self, n=1, c=32, h=30, w=40):
        image = torch.randn(size=(n, c, h, w), dtype=torch.float32)
        disparity = torch.randn(size=(n, 1, h, w), dtype=torch.float32) * w
        return (image, disparity)

    def get_output_number(self):
        return 1

    def warp_by_flow_map(self, img, flow):
        """
        warp image according to stereo flow map (i.e. disparity map)

        Args:
            [1] img, N x C x H x W, original image or feature map
            [2] flow,  N x 1 x H x W or N x 2 x H x W, flow map

        Return:
            [1] warped, N x C x H x W, warped image or feature map
        """
        n, c, h, w = flow.shape

        assert c == 1 or c == 2, f"invalid flow map dimension 1 or 2 ({c})!"

        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=img.device, dtype=img.dtype),
            torch.arange(w, device=img.device, dtype=img.dtype),
            indexing="ij",
        )

        grid_x = grid_x.view([1, 1, h, w]) - flow[:, 0, :, :].view([n, 1, h, w])
        grid_x = grid_x.permute([0, 2, 3, 1])

        if c == 2:
            grid_y = grid_y.view([1, 1, h, w]) - flow[:, 1, :, :].view([n, 1, h, w])
            grid_y = grid_y.permute([0, 2, 3, 1])
        else:
            grid_y = grid_y.view([1, h, w, 1]).repeat(n, 1, 1, 1)

        grid_x = 2.0 * grid_x / (w - 1.0) - 1.0
        grid_y = 2.0 * grid_y / (h - 1.0) - 1.0
        grid_map = torch.concatenate((grid_x, grid_y), dim=-1)

        warped = F.grid_sample(
            img,
            grid_map,
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )
        return warped

    def process_grid_coordinates(self, pixs, width):
        # if self.align_corners:
        #     pixels = (grids + 1.0) * (0.5 * (width - 1))
        # else:
        #     pixels = (grids + 1.0) * (0.5 * width) - 0.5
        assert self.align_corners

        if self.padding_mode == "border":
            pixs = torch.clip(pixs, 0, width - 1)
        elif self.padding_mode == "zeros":
            pixs = torch.clip(pixs, -1, width) + 1
        elif self.padding_mode == "reflection":
            if self.align_corners:
                pixs = (width - 1) - torch.abs(
                    pixs % (2 * max(width - 1, 1)) - (width - 1)
                )
            else:
                pixs = width - torch.abs((pixs + 0.5) % (2 * width) - width) - 0.5
                pixs = torch.clip(pixs, 0, width - 1)
        return pixs

    def warp_by_disparity_map(self, img, disp):
        """
        reference: https://github.com/AlexanderLutsenko/nobuco/blob/aa4745e6abb1124d90f7d3ace6d282f923f08a40/nobuco/node_converters/grid_sampling.py#L38
        """
        n, _, h, w = img.shape

        x = torch.arange(w, device=img.device, dtype=img.dtype)
        x = x[None, None, None, :].repeat([n, 1, h, 1])
        x = x - disp

        x = self.process_grid_coordinates(x, w)

        if self.padding_mode == "zeros":
            x = F.pad(x, (1, 1, 0, 0), mode="constant", value=0)
            img = F.pad(img, (1, 1, 0, 0), mode="constant", value=0)

        if self.mode == "bilinear":
            x0 = torch.floor(x).type(dtype=torch.int64)
            x1 = torch.ceil(x).type(dtype=torch.int64)

            dx = x - x0

            v_x0 = torch.gather(img, dim=-1, index=x0.expand_as(img))
            v_x1 = torch.gather(img, dim=-1, index=x1.expand_as(img))

            warped = (1.0 - dx) * v_x0 + dx * v_x1

        elif self.mode == "nearest":
            x0 = torch.round(x).type(dtype=torch.int64)
            warped = torch.gather(img, dim=-1, index=x0.expand_as(img))
        else:
            raise NotImplementedError

        if self.padding_mode == "zeros":
            warped = warped[:, :, :, 1:-1]
        return warped

    def forward(self, img, disp):
        return self.warp_by_disparity_map(img, disp)


class CostVolume2D(nn.Module):
    def __init__(self, max_disp, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.max_disp = max_disp

        self.l_kernels = self.get_conv2d_kernels(reverse=False)
        self.r_kernels = self.get_conv2d_kernels(reverse=True)

    def get_conv2d_kernels(self, reverse=True):
        kernels = []
        for i in range(1, self.max_disp):
            kernel = np.zeros(shape=[1, 1, 1, i + 1], dtype=np.float32)
            kernel[0, 0, 0, (0 if reverse else -1)] = 1.0
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
        c = l_fmap.shape[1]

        cost_volume = []
        for d in range(self.max_disp):
            if d == 0:
                cost_volume.append(self.get_cost_item(l_fmap, r_fmap))
            else:
                x = F.conv2d(
                    l_fmap,
                    self.l_kernels[d - 1].to(l_fmap.device).repeat([c, 1, 1, 1]),
                    stride=1,
                    padding=0,
                    groups=c,
                )
                y = F.conv2d(
                    r_fmap,
                    self.r_kernels[d - 1].to(r_fmap.device).repeat([c, 1, 1, 1]),
                    stride=1,
                    padding=0,
                    groups=c,
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


class DisparityRegress(nn.Module):
    def __init__(
        self, hidden_dims=[128, 128, 96, 64, 32], max_disp=2, out_dim=1, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.hidden_dims = hidden_dims
        self.max_disp = max_disp
        self.in_dim = max_disp + 1
        self.out_dim = out_dim

        self.warp_header = Warp1DOp(
            mode="bilinear", padding_mode="border", align_corners=True
        )
        self.cost_builder = CostVolume2D(max_disp=max_disp)

        conv_layers = nn.ModuleList([])
        for i in range(len(hidden_dims) + 1):
            conv_layers.append(
                BasicConv(
                    in_channels=self.in_dim if i == 0 else hidden_dims[i - 1],
                    out_channels=hidden_dims[i] if i < len(hidden_dims) else out_dim,
                    deconv=False,
                    is_3d=False,
                    bn=True,
                    relu=True,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
            )
        self.conv_layers = nn.Sequential(*conv_layers)

    def forward(self, l_fmap, r_fmap, l_disp_down, is_train):
        if l_disp_down is None:
            n, _, h, w = l_fmap.shape

            l_disp_up = torch.zeros(
                size=[n, 1, h, w], dtype=torch.float32, device=l_fmap.device
            )

            r_fmap_warp = r_fmap
        else:
            l_disp_up = (
                F.interpolate(
                    l_disp_down,
                    scale_factor=2.0,
                    mode="bilinear",
                    align_corners=True,
                )
                * 2.0
            )
            r_fmap_warp = self.warp_header(r_fmap, l_disp_up)
        cost_volume = self.cost_builder(l_fmap, r_fmap_warp, is_train)
        cost_volume = torch.concat((cost_volume, l_disp_up), dim=1)
        l_disp_up = self.conv_layers(cost_volume)
        return l_disp_up


class DisparityRefine(nn.Module):
    def __init__(
        self,
        unify_dim=64,
        refine_dims=[128, 128, 96, 64, 32, 1],
        refine_dilates=[1, 2, 4, 8, 1, 1],
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.unify_dim = unify_dim
        self.refine_dims = refine_dims
        self.refine_dilates = refine_dilates

        assert len(self.refine_dims) == len(self.refine_dilates)

        conv_refine_layers = nn.ModuleList([])
        for i in range(len(refine_dims)):
            conv_refine_layers.append(
                BasicConv(
                    in_channels=(unify_dim + 1) if i == 0 else refine_dims[i - 1],
                    out_channels=refine_dims[i],
                    deconv=False,
                    is_3d=False,
                    bn=True,
                    relu=True,
                    kernel_size=3,
                    stride=1,
                    padding=refine_dilates[i],
                    dilation=refine_dilates[i],
                )
            )
        self.conv_refine = nn.Sequential(*conv_refine_layers)

    def forward(self, l_disp, l_fmap):
        return self.conv_refine(torch.concat((l_disp, l_fmap), dim=1))


class FastMADNet(nn.Module):
    def __init__(
        self,
        in_dim=3,
        hidden_dims=[16, 32, 48, 64, 96, 128],
        regress_dims=[64, 64, 48, 32, 16],
        max_disp=2,
        unify_dim=64,
        refine_dims=[64, 64, 48, 32, 16, 1],
        refine_dilates=[1, 2, 4, 8, 1, 1],
        use_cascade=True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.regress_dims = regress_dims
        self.max_disp = max_disp
        self.unify_dim = unify_dim
        self.refine_dims = refine_dims
        self.refine_dilates = refine_dilates
        self.use_cascade = use_cascade

        self.feature_extractor = FeatureExtract(
            in_dim=in_dim, hidden_dims=hidden_dims, unify_dim=unify_dim
        )

        self.disparity_regressor = DisparityRegress(
            hidden_dims=regress_dims, max_disp=max_disp, out_dim=1
        )
        self.disparity_refiner = DisparityRefine(
            unify_dim=unify_dim,
            refine_dims=refine_dims,
            refine_dilates=refine_dilates,
        )

    def forward(self, l_img, r_img, is_train):
        l_fmaps = self.feature_extractor(l_img)
        r_fmaps = self.feature_extractor(r_img)

        l_disp = None
        l_disps = []
        for i in range(len(l_fmaps)):
            j = len(l_fmaps) - 1 - i

            l_disp = self.disparity_regressor(l_fmaps[j], r_fmaps[j], l_disp, is_train)

            l_disp_refine = self.disparity_refiner(l_disp, l_fmaps[j])

            if is_train or i == len(l_fmaps) - 1:
                l_disps.append(l_disp_refine)
            
            if self.use_cascade:
                l_disp = l_disp_refine
        
        if is_train:
            return l_disps, l_fmaps, r_fmaps
        else:
            return l_disps


from tools.profiler import get_model_capacity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FastMADNet().to(device)
sample = torch.rand(size=(1, 3, 448, 640), dtype=torch.float32).to(device)
_ = get_model_capacity(module=model, inputs=(sample, sample, False), verbose=True)
