from __future__ import print_function
import math
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from others.fast_acv_net.submodule import Conv2x, BasicConv


class ModuleWithInit(nn.Module):
    def __init__(self):
        super(ModuleWithInit, self).__init__()

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


class MobileNetV2(ModuleWithInit):
    def __init__(self, *args, **kwargs):
        super(MobileNetV2, self).__init__(*args, **kwargs)
        model = timm.create_model(
            "mobilenetv2_100", pretrained=True, features_only=True
        )
        self.layer_idxs = [1, 2, 3, 5, 6]
        self.hidden_dims = [16, 24, 32, 96, 160]

        self.down_factor = 5

        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1
        self.act1 = nn.ReLU()

        self.block0 = torch.nn.Sequential(*model.blocks[0 : self.layer_idxs[0]])
        self.block1 = torch.nn.Sequential(
            *model.blocks[self.layer_idxs[0] : self.layer_idxs[1]]
        )
        self.block2 = torch.nn.Sequential(
            *model.blocks[self.layer_idxs[1] : self.layer_idxs[2]]
        )
        self.block3 = torch.nn.Sequential(
            *model.blocks[self.layer_idxs[2] : self.layer_idxs[3]]
        )
        self.block4 = torch.nn.Sequential(
            *model.blocks[self.layer_idxs[3] : self.layer_idxs[4]]
        )

    def forward(self, x):
        x = self.act1(self.bn1(self.conv_stem(x)))
        x2 = self.block0(x)
        x4 = self.block1(x2)
        x8 = self.block2(x4)
        x16 = self.block3(x8)
        x32 = self.block4(x16)
        return [x4, x8, x16, x32]


class FeatureUpsample(ModuleWithInit):
    def __init__(self, hidden_dims=[24, 32, 96, 160], *args, **kwargs):
        super(FeatureUpsample, self).__init__()
        self.hidden_dims = hidden_dims

        self.deconv_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            j = len(hidden_dims) - 1 - i

            self.deconv_layers.append(
                Conv2x(
                    hidden_dims[j] * (1 if i == 0 else 2),
                    hidden_dims[j - 1],
                    deconv=True,
                    concat=True,
                )
            )

        self.conv_out = BasicConv(
            hidden_dims[0] * 2, hidden_dims[0] * 2, kernel_size=3, stride=1, padding=1
        )

        self.weight_init()

    def forward(self, x_pyramid):
        for i in range(len(x_pyramid) - 1):
            j = len(x_pyramid) - 1 - i

            # e.g. MobileNetV2-100 -> [x4, x8, x16, x32]
            #   x16 = DECONV[0](x32, x16)
            #   x8  = DECONV[1](x16, x8)
            #   x4  = DECONV[2](x8, x4)
            #   x4  = CONV_OUT(x4)
            x_pyramid[j - 1] = self.deconv_layers[i](x_pyramid[j], x_pyramid[j - 1])

        x_pyramid[0] = self.conv_out(x_pyramid[0])
        return x_pyramid


class FeatureStem(ModuleWithInit):
    def __init__(
        self, in_dim=3, stem_hidden_dims=[32, 48, 48], stem_out_dim=32, *args, **kwargs
    ):
        super(FeatureStem, self).__init__(*args, **kwargs)
        self.in_dim = in_dim
        self.stem_hidden_dims = stem_hidden_dims
        self.stem_out_dim = stem_out_dim

        self.conv_layers = nn.ModuleList([])
        for i in range(len(stem_hidden_dims)):
            self.conv_layers.append(
                nn.Sequential(
                    BasicConv(
                        in_dim if i == 0 else stem_hidden_dims[i - 1],
                        stem_hidden_dims[i],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.Conv2d(
                        stem_hidden_dims[i],
                        stem_hidden_dims[i]
                        if i != len(stem_hidden_dims) - 1
                        else stem_out_dim,
                        3,
                        1,
                        1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(
                        stem_hidden_dims[i]
                        if i != len(stem_hidden_dims) - 1
                        else stem_out_dim
                    ),
                    nn.ReLU(),
                )
            )

        self.weight_init()

    def forward(self, x):
        x_pyramid = []

        for conv_layer in self.conv_layers:
            x = conv_layer(x)

            x_pyramid.append(x)
        return x_pyramid


class MobileV2FeatureExtract(nn.Module):
    def __init__(
        self, in_dim=3, stem_hidden_dims=[32, 48, 48], stem_out_dim=32, *args, **kwargs
    ):
        super(MobileV2FeatureExtract, self).__init__(*args, **kwargs)

        self.backbone = MobileNetV2()
        self.backbone_hidden_dims = self.backbone.hidden_dims
        self.upsample = FeatureUpsample(hidden_dims=self.backbone_hidden_dims[1:])

        self.stem_hidden_dims = stem_hidden_dims
        self.stem_out_dim = stem_out_dim
        self.stem = FeatureStem(
            in_dim=in_dim, stem_hidden_dims=stem_hidden_dims, stem_out_dim=stem_out_dim
        )

    def forward(self, x):
        # mobile_fmaps = [x4, x8, x16, x32]
        mobile_fmaps = self.upsample(self.backbone(x))

        # stem_fmaps = [x2, x4, x8]
        stem_fmaps = self.stem(x)

        x4 = torch.concat((mobile_fmaps[0], stem_fmaps[1]), dim=1)
        x8 = torch.concat((mobile_fmaps[1], stem_fmaps[2]), dim=1)
        return [x4, x8]


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
        cost_item = (l_fmap * r_fmap).mean(dim=2)
        return cost_item

    def make_cost_volume_naive(self, l_fmap, r_fmap):
        # left: 1 x 32 x 60 x 80
        # right: 1 x 32 x 60 x 80
        # max_disp: 24
        # cost_volume: 1 x 32 x 24 x 60 x 80
        n, c, h, w = l_fmap.shape

        cost_volume = torch.ones(
            size=[n, self.num_cost_groups, self.max_disp, h, w], dtype=l_fmap.dtype, device=l_fmap.device
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
                    F.pad(cost_item, [d, 0, 0, 0], value=1.0).unsqueeze(2)
                )
        cost_volume = torch.concat(cost_volume, dim=2)
        return cost_volume

    def forward(self, l_fmap, r_fmap, use_naive=True):
        return (
            self.make_cost_volume_naive(l_fmap, r_fmap)
            if use_naive
            else self.make_cost_volume_conv2d(l_fmap, r_fmap)
        )


class MobileStereoNetV6(nn.Module):
    def __init__(
        self,
        in_dim=3,
        stem_hidden_dims=[32, 48, 48],
        stem_out_dim=32,
        max_disp=192,
        num_cost_groups=12,
        use_concat_volume=True,
    ):
        super(MobileStereoNetV6, self).__init__()
        self.max_disp = max_disp
        self.num_cost_groups = num_cost_groups
        self.use_concat_volume = use_concat_volume
        self.feature_extractor = MobileV2FeatureExtract(
            in_dim=in_dim, stem_hidden_dims=stem_hidden_dims, stem_out_dim=stem_out_dim
        )

        self.warp_head = Warp1DOp(
            mode="bilinear", padding_mode="border", align_corners=True
        )

        self.cost_builder = GroupwiseCostVolume3D(
            hidden_dim=self.feature_extractor.backbone_hidden_dims[2],
            max_disp=self.max_disp // 8,
            num_cost_groups=num_cost_groups,
        )

    def concat_volume_generator(self, l_fmap, r_fmap, l_disp):
        r_fmap_warp = self.warp_head(r_fmap, l_disp)
        concat_volume = torch.cat((l_fmap, r_fmap_warp), dim=1)
        return concat_volume

    def forward(self, l_img, r_img, is_train=True):
        l_fmap_4x, l_fmap_8x = self.feature_extractor(l_img)
        r_fmap_4x, r_fmap_8x = self.feature_extractor(r_img)

        # cost_volume: 1 x 12 x 24 x 60 x 80, 8x
        cost_volume = self.cost_builder(l_fmap_8x, r_fmap_8x)

        print()