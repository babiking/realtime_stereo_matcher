import torch
import torch.nn as nn


class TorchGroupwiseCost(nn.Module):
    def __init__(self, n_groups, max_disparity, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.n_groups = n_groups
        self.max_disparity = max_disparity

    def groupwise(self, left, right, n_groups):
        n, c, h, w = left.shape

        assert (
            c % n_groups == 0
        ), f"groupwise cost channel ({c}) % #groups ({n_groups}) != 0."

        c_per_group = c // n_groups

        cost = (left * right).view([n, n_groups, c_per_group, h, w]).mean(dim=2)
        return cost

    def forward(self, left, right):
        """
        torch naive groupwise product of left and right image features to generate cost volume

        Args:
            [1] left, (N, C, H, W), left image features
            [2] right, (N, C, H, W), right image features
            [3] n_groups, Scalar/Int, number of groups for each disparity level
            [4] max_disparity, Scalar/Int, maximum disparity value

        Return:
            [1] volume, (N, C', H, W, D'), i.e. (N, n_groups, H, W, max_disparity), cost volume
        """
        n, c, h, w = left.shape

        volume = torch.zeros([n, self.n_groups, h, w, self.max_disparity])

        for i in range(self.max_disparity):
            if i == 0:
                volume[:, :, :, :, i] = self.groupwise(
                    left,
                    right,
                    self.n_groups,
                )
            else:
                volume[:, :, :, i:, i] = self.groupwise(
                    left[:, :, :, i:],
                    right[:, :, :, :-i],
                    self.n_groups,
                )

        volume = volume.contiguous()
        return volume
