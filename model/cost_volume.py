import torch
import torch.nn as nn
import torch.nn.functional as F


def cost_volume_factory(config):
    cost_type = config["type"]
    if cost_type == "difference_3d":
        return DifferenceCostVolume3D(**config["arguments"])
    elif cost_type == "groupwise_3d":
        return GroupwiseCostVolume3D(**config["arguments"])
    elif cost_type == "concat_3d":
        return ConcatCostVolume3D(**config["arguments"])
    else:
        raise NotImplementedError(f"invalid cost volume type: {cost_type}!")


class BaseCostVolume3D(nn.Module):

    def __init__(self, max_disp, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.max_disp = max_disp

    def init_cost_volume(self, left, right):
        n, c, h, w = left.shape

        volume = torch.zeros(size=[n, c, self.max_disp, h, w],
                             dtype=left.dtype,
                             device=left.device)
        return volume

    def get_pairwise_cost(self, left, right):
        raise NotImplementedError

    def forward(self, left, right):
        """
            BaseCostVolume to compute epipolar pix2pix correlation between stereo feature maps.

            Args:
                [1] left:  N x C x H x W
                [2] right: N x C x H x W
                [3] max_disp: Scalar, max disparity value

            Return:
                [1] cost_volume: N x C x D x H x W
        """
        volume = self.init_cost_volume(left, right)
        for d in range(self.max_disp):
            if d == 0:
                volume[:, :, d, :, :] = self.get_pairwise_cost(left, right)
            else:
                volume[:, :, d, :, d:] = \
                    self.get_pairwise_cost(left[:, :, :, d:], right[:, :, :, :-d])
        volume = volume.contiguous()
        return volume


class DifferenceCostVolume3D(BaseCostVolume3D):

    def __init__(self, max_disp, *args, **kwargs) -> None:
        super().__init__(max_disp, *args, **kwargs)

    def init_cost_volume(self, left, right):
        n, c, h, w = left.shape

        volume = torch.ones(size=[n, c, self.max_disp, h, w],
                            dtype=left.dtype,
                            device=left.device)
        return volume

    def get_pairwise_cost(self, left, right):
        cost = left - right
        return cost


class GroupwiseCostVolume3D(BaseCostVolume3D):

    def __init__(self, max_disp, num_groups, *args, **kwargs) -> None:
        super().__init__(max_disp, *args, **kwargs)

        self.num_groups = num_groups

    def get_pairwise_cost(self, left, right):
        n, c, h, w = left.shape

        assert c % self.num_groups == 0, \
            f"for GWC cost volume, #channels({c}) should be evenly divided by #groups({self.num_groups})."

        cost = \
            (left * right).view([n, self.num_groups, c // self.num_groups, h, w]).mean(dim=2)
        return cost


class ConcatCostVolume3D(BaseCostVolume3D):

    def __init__(self, max_disp, *args, **kwargs) -> None:
        super().__init__(max_disp, *args, **kwargs)

    def init_cost_volume(self, left, right):
        n, c, h, w = left.shape

        volume = torch.ones(size=[n, 2 * c, self.max_disp, h, w],
                            dtype=left.dtype,
                            device=left.device)
        return volume

    def get_pairwise_cost(self, left, right):
        cost = torch.concat([left, right], dim=1)
        return cost

    def forward(self, left, right):
        n, c, h, w = left.shape

        volume = self.init_cost_volume(left, right)
        for d in range(self.max_disp):
            if d == 0:
                volume[:, :, d, :, :] = self.get_pairwise_cost(left, right)
            else:
                volume[:, :c, d, :, :] = left
                volume[:, c:, d, :, d:] = right[:, :, :, :-d]
        volume = volume.contiguous()
        return volume
