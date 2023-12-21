import torch
import torch.nn as nn


class TorchConv3DCostAggregate(nn.Module):
    def __init__(
        self,
        in_dim,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        n_aggregates=4,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.in_dim = in_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.n_aggregates = n_aggregates

        self.aggregate_layers = nn.ModuleList()
        for _ in range(n_aggregates):
            self.aggregate_layers.append(
                self.conv3d(
                    in_dim,
                    in_dim,
                    kernel_size,
                    stride,
                    dilation,
                    groups,
                )
            )
        self.aggregate_layers = nn.Sequential(*self.aggregate_layers)

        self.final_conv3d = nn.Conv3d(
            in_dim, 1, kernel_size=3, stride=1, padding=1, bias=True
        )

    @staticmethod
    def conv3d(
        in_dim,
        out_dim,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        leaky_relu_ratio=0.2,
    ):
        return nn.Sequential(
            nn.Conv3d(
                in_dim,
                out_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=dilation,
                dilation=dilation,
                bias=False,
                groups=groups,
            ),
            nn.BatchNorm3d(out_dim),
            nn.LeakyReLU(leaky_relu_ratio, inplace=True),
        )

    def forward(self, cost_volume):
        # cost_volume, [B, C, D, H, W]
        assert (
            cost_volume.dim() == 5
        ), f"cost volume #dimensions != 5 ({cost_volume.dim()})."

        out_cost_volume = self.aggregate_layers(cost_volume)
        # cost_volume: [B, C, D, H, W] -> [B, 1, D, H, W]
        out_cost_volume = self.final_conv3d(out_cost_volume)
        # cost_volume: [B, 1, D, H, W] -> [B, D, H, W]
        out_cost_volume = out_cost_volume.squeeze(1)
        return out_cost_volume
