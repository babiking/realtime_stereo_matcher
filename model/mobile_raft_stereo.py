import torch
import torch.nn as nn
import torch.nn.functional as F
from feature_extractor.residual_block import MobileV2Residual, ResidualBottleneckBlock
from cost_volume.inner_product import TorchInnerProductCost
from flow_update.gru_update import (
    MySoftArgminFlowHead,
    MyGRUFlowMapUpdata,
    MyStereoEdgeHead,
)


class ImageFeatureEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dims, block="mobilev2", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.in_dim = in_dim
        self.hidden_dims = hidden_dims

        self.layers = nn.ModuleList([])
        layer0 = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dims[0], kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_dims[0], hidden_dims[0], kernel_size=1, stride=1, padding=0
            ),
            nn.ReLU(inplace=True),
        )
        self.layers.append(layer0)

        for i in range(1, len(hidden_dims)):
            if block == "mobilev2":
                layeri = MobileV2Residual(
                    in_dim=hidden_dims[i - 1],
                    out_dim=hidden_dims[i],
                    stride=2,
                    expanse_ratio=1,
                    dilation=2,
                )
            elif block == "bottleneck":
                layeri = ResidualBottleneckBlock(
                    in_dim=hidden_dims[i - 1],
                    out_dim=hidden_dims[i],
                    norm_fn="group",
                    stride=2,
                )
            self.layers.append(layeri)

    def forward(self, left, right):
        left = (2.0 * (left / 255.0) - 1.0).contiguous()
        right = (2.0 * (right / 255.0) - 1.0).contiguous()

        # 2 x 3 x 480 x 640
        x = torch.concatenate((left, right), dim=0)

        pyramid = []
        for layer in self.layers:
            x = layer(x)
            pyramid.append(x)
        return pyramid


def warp_feature_volume_by_flow(feature_volume, flow_map):
    n, c, h, w = feature_volume.shape

    ys, xs = torch.meshgrid(
        torch.arange(h, device=flow_map.device), torch.arange(w, device=flow_map.device)
    )
    xs = flow_map.permute([0, 2, 3, 1]) + xs.view([1, h, w, 1])
    ys = ys.view([1, h, w, 1]).repeat(n, 1, 1, 1)

    grid_map = torch.concatenate([2.0 * xs / w - 1.0, 2.0 * ys / h - 1.0], dim=-1)

    feature_warp = F.grid_sample(
        feature_volume.float(), grid_map, padding_mode="zeros", align_corners=True
    )
    return feature_warp


class MobileRaftStereoModel(nn.Module):
    def __init__(
        self,
        hidden_dim,
        encode_block,
        down_factor,
        cost_factor,
        cost_radius,
        max_disparity,
        mixed_precision,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.hidden_dim = hidden_dim
        self.encode_block = encode_block
        self.down_factor = down_factor
        self.down_scale = 2 ** (down_factor - 1)
        self.cost_factor = cost_factor
        self.cost_radius = cost_radius
        self.max_disparity = max_disparity
        self.mixed_precision = mixed_precision

        self.encoder = ImageFeatureEncoder(
            in_dim=3, hidden_dims=[hidden_dim] * down_factor, block=self.encode_block
        )

        self.cost_volume_builder = TorchInnerProductCost(
            max_disparity=max_disparity // self.down_scale
        )
        self.flow_map_header = MySoftArgminFlowHead(
            max_disparity=max_disparity // self.down_scale
        )

        self.flow_map_update = MyGRUFlowMapUpdata(
            in_dim=max_disparity // self.down_scale,
            hidden_dim=hidden_dim,
            num_of_updates=cost_factor,
        )

        self.edge_map_header = MyStereoEdgeHead(
            in_dim=hidden_dim, hidden_dims=[hidden_dim] * 3
        )

    def forward(self, left, right):
        encode_pyramid = self.encoder(left, right)

        l_fmap, r_fmap = torch.split(
            encode_pyramid[-1],
            split_size_or_sections=encode_pyramid[-1].shape[0] // 2,
            dim=0,
        )
        cost_volume = self.cost_volume_builder(l_fmap, r_fmap)
        init_flow_map = self.flow_map_header(cost_volume)
        delta_flow_map = self.flow_map_update(cost_volume)

        flow_map = init_flow_map + delta_flow_map
        flow_map = F.interpolate(
            flow_map, scale_factor=self.down_scale, mode="bilinear", align_corners=True
        )
        flow_map *= self.down_scale

        # flow_pyramid = [flow_map]
        # for j in np.flip(range(self.down_factor - 1)):
        #     flow_map = F.interpolate(
        #         flow_map, scale_factor=[2, 2], mode="bilinear", align_corners=True
        #     ) * 2.0

        #     l_fmap, r_fmap = torch.split(
        #         encode_pyramid[j],
        #         split_size_or_sections=encode_pyramid[j].shape[0] // 2,
        #         dim=0,
        #     )
        #     r_warp_fmap = warp_feature_volume_by_flow(r_fmap, flow_map)

        #     edge_map = self.edge_map_header(l_fmap, r_warp_fmap)

        #     flow_map += edge_map

        #     flow_pyramid.append(flow_map)

        # return flow_pyramid

        return [flow_map * -1.0]
