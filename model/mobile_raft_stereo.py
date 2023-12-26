import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from feature_extractor.residual_block import ResidualBottleneckBlock
from cost_volume.inner_product import TorchInnerProductCost
from cost_sample.grid_sample import TorchGridSampleSearch
from flow_update.gru_update import (
    MySoftArgminFlowHead,
    MyGRUFlowMapUpdata,
    MyStereoEdgeHead,
)


class ResidualBottleneckEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dims, *args, **kwargs) -> None:
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
        down_factor,
        cost_factor,
        cost_radius,
        mixed_precision,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.hidden_dim = hidden_dim
        self.down_factor = down_factor
        self.cost_factor = cost_factor
        self.cost_radius = cost_radius
        self.mixed_precision = mixed_precision

        self.encoder = ResidualBottleneckEncoder(
            in_dim=3, hidden_dims=[hidden_dim] * down_factor
        )

        self.cost_volume_builder = TorchInnerProductCost()
        self.cost_volume_sampler = TorchGridSampleSearch(search_range=cost_radius)

        self.flow_map_update = MyGRUFlowMapUpdata(
            in_dim=2 * cost_radius + 1, hidden_dim=32
        )
        self.flow_map_header = MySoftArgminFlowHead()
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
        cost_volume = torch.unsqueeze(cost_volume, dim=1)

        init_flow_map = None
        cost_pyramid = []
        for i in range(self.cost_factor):
            if i > 0:
                cost_volume = F.avg_pool3d(cost_volume, kernel_size=2, stride=2)

            n, _, h, w, d = cost_volume.shape

            flow_map = self.flow_map_header(torch.squeeze(cost_volume, dim=1))

            if i == 0:
                init_flow_map = flow_map.clone()

            cost_sample = self.cost_volume_sampler(
                cost_volume.view([n, 1, h * w, d]), flow_map
            )
            cost_pyramid.append(cost_sample)
        delta_flow_map = self.flow_map_update(cost_pyramid)

        flow_map = init_flow_map.permute([0, 3, 1, 2]) + delta_flow_map

        flow_pyramid = [flow_map]
        for j in np.flip(range(self.down_factor - 1)):
            flow_map = F.interpolate(
                flow_map, scale_factor=[2, 2], mode="bilinear", align_corners=True
            )

            l_fmap, r_fmap = torch.split(
                encode_pyramid[j],
                split_size_or_sections=encode_pyramid[j].shape[0] // 2,
                dim=0,
            )
            r_warp_fmap = warp_feature_volume_by_flow(r_fmap, flow_map)

            edge_map = self.edge_map_header(l_fmap, r_warp_fmap)

            flow_map += edge_map

            flow_pyramid.append(flow_map)

        return flow_pyramid
