import torch
import torch.nn as nn
import torch.nn.functional as F


class TorchGridSampleSearch(nn.Module):
    def __init__(self, search_range, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.search_range = search_range
        self.search_linespace = torch.linspace(
            -search_range, search_range, 2 * search_range + 1, dtype=torch.float32
        )

    def forward(self, cost_volume, flow_map):
        """
        torch naive grid sample SEARCH of cost volume based on flow map

        Args:
            [1] cost_volume, (N, 1, H * W, D)
            [2] flow_map, (N, H, W, 1), disparity flow map, s.t. -1.0 < flow_map < 1.0
        Return:
            [1] search_cost_volume, (N, 2 * search_range + 1, H, W)
        """
        n, c, _, d = cost_volume.shape
        _, h, w, _ = flow_map.shape

        assert c == 1, f"TorchGridSampleSearch input feature dimension != 1 ({c})."

        flow_xs = flow_map + self.search_linespace
        flow_xs = flow_xs.view([n * h * w, (2 * self.search_range + 1), 1, 1])

        flow_ys = torch.zeros_like(flow_xs)

        # flow_grids: (N * H * W, 2 * search_range + 1, 1, 2)
        flow_grids = torch.concatenate([2.0 * flow_xs / d - 1.0, flow_ys], dim=-1)

        all_cost_volume = cost_volume.view([n * h * w, 1, 1, d])

        # search_cost_volume: (N * H * W, 1, 2 * search_range + 1, 1)
        search_cost_volume = F.grid_sample(
            all_cost_volume,
            flow_grids,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        search_cost_volume = torch.squeeze(search_cost_volume)
        search_cost_volume = search_cost_volume.view(
            [n, h, w, (2 * self.search_range + 1)]
        )
        search_cost_volume = search_cost_volume.permute([0, 3, 1, 2])
        return search_cost_volume

    def __str__(self) -> str:
        return f"{self.__class__.__name__} | search={self.search_range}"


class TorchGridSampleParse(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, cost_volume, flow_map):
        """
        torch naive grid sample PARSE of cost volume based on flow map

        Args:
            [1] cost_volume, (N, C, H * W, D)
            [2] flow_map, (N, H, W, 1), disparity flow map, s.t. -1.0 < flow_map < 1.0
        Return:
            [1] sample_cost_volume, (N, C, H, W)
        """
        n, c, _, d = cost_volume.shape
        _, h, w, _ = flow_map.shape

        flow_xs = flow_map.view([n * h * w, 1, 1, 1])
        flow_ys = torch.zeros_like(flow_xs)

        # flow_grids: (N * H * W, 1, 1, 2)
        flow_grids = torch.concatenate([2.0 * flow_xs / d - 1.0, flow_ys], dim=-1)

        all_cost_volume = cost_volume.permute([0, 2, 1, 3]).view([n * h * w, c, 1, d])

        # search_cost_volume: (N * H * W, C, 1, 1)
        parse_cost_volume = F.grid_sample(
            all_cost_volume,
            flow_grids,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

        parse_cost_volume = torch.squeeze(parse_cost_volume)
        parse_cost_volume = parse_cost_volume.view([n, h, w, c])
        parse_cost_volume = parse_cost_volume.permute([0, 3, 1, 2])
        return parse_cost_volume


class MyGridSampleParse(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, cost_volume, flow_map):
        """
        customize grid sample PARSE of cost volume based on flow map

        Args:
            [1] cost_volume, (N, C, H * W, D)
            [2] flow_map, (N, H, W, 1), disparity flow map, s.t. -1.0 < flow_map < 1.0
        Return:
            [1] sample_cost_volume, (N, C, H, W)
        """
        raise NotImplementedError
