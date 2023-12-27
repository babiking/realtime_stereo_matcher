import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MySoftArgminFlowHead(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, cost_volume):
        """
        customize soft argmin flow map generation head.

        Args:
            [1] cost_volume, (N, H, W, D), cost volume

        Return:
            [1] flow_map, (N, H, W, D), flow map by soft-argmin along disparity dimension
        """
        n, h, w, d = cost_volume.shape

        disparities = torch.linspace(0, d, d, dtype=torch.float32, device=cost_volume.device)
        disparities = disparities.view([1, 1, 1, d])

        flow_map = torch.sum(
            torch.softmax(cost_volume, dim=-1) * disparities, dim=-1, keepdim=True
        )
        return flow_map


class MyStereoEdgeHead(nn.Module):
    def __init__(self, in_dim, hidden_dims, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.conv2d_layers = nn.ModuleList([])

        for i in range(len(hidden_dims) + 1):
            if i == 0:
                kernel_size = 7
                in_channels = in_dim
                out_channels = hidden_dims[0]
                dilation = 1
            elif i == len(hidden_dims):
                kernel_size = 1
                in_channels = hidden_dims[len(hidden_dims) - 1]
                out_channels = 1
                dilation = 1
            else:
                kernel_size = 3
                in_channels = hidden_dims[i - 1]
                out_channels = hidden_dims[i]
                dilation = 2

        self.conv2d_layers.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding="same",
                    dilation=dilation,
                    groups=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        )

    def forward(self, left, right):
        """
        generate edge map from stereo cost volume

        Args:
            [1] left, (N, C, H, W), left image feature
            [2] right, (N, C, H, W), right image feature

        Return:
            [1] edge_map, (N, 1, H, W), predicted edge map
        """
        edge_map = left - right

        for conv2d in self.conv2d_layers:
            edge_map = conv2d(edge_map)
        return edge_map


class MyGRUCell(nn.Module):
    def __init__(self, in_dim, hidden_dim, kernel_size=3, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.conv_xz = nn.Conv2d(
            in_dim, hidden_dim, kernel_size, padding=kernel_size // 2
        )
        self.conv_hz = nn.ConvTranspose2d(
            hidden_dim,
            hidden_dim,
            kernel_size,
            stride=2,
            padding=kernel_size // 2,
            output_padding=1,
        )

        self.conv_xr = nn.Conv2d(
            in_dim, hidden_dim, kernel_size, padding=kernel_size // 2
        )
        self.conv_hr = nn.ConvTranspose2d(
            hidden_dim,
            hidden_dim,
            kernel_size,
            stride=2,
            padding=kernel_size // 2,
            output_padding=1,
        )

        self.conv_xq = nn.Conv2d(
            in_dim, hidden_dim, kernel_size, padding=kernel_size // 2
        )
        self.conv_hq = nn.ConvTranspose2d(
            hidden_dim,
            hidden_dim,
            kernel_size,
            stride=2,
            padding=kernel_size // 2,
            output_padding=1,
        )

        self.conv_hh = nn.ConvTranspose2d(
            hidden_dim,
            hidden_dim,
            kernel_size,
            stride=2,
            padding=kernel_size // 2,
            output_padding=1,
        )

    def forward(self, x, h):
        """
        GRUCell:
            Z(t) = sigmoid(W_xz * X(t) + W_hz * H(t - 1))
            R(t) = sigmoid(W_xr * X(t) + W_hr * H(t - 1))
            H~(t) = tanh(W_xh * X(t) + R(t) .* (W_hh * H(t - 1)))
            H(t) = (1 - Z(t)) .* H~(t) + Z(t) .* H(t - 1)

        """
        z = torch.sigmoid(self.conv_xz(x) + self.conv_hz(h))

        r = torch.sigmoid(self.conv_xr(x) + self.conv_hr(h))

        q = torch.tanh(self.conv_xq(x) + r * self.conv_hq(h))

        h = (1.0 - z) * q + z * self.conv_hh(h)
        return h


class MyGRUFlowMapUpdata(nn.Module):
    def __init__(self, in_dim, hidden_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.hidden_dim = hidden_dim

        self.gru_cell = MyGRUCell(in_dim, hidden_dim, kernel_size=3)

        self.out_1x1_layer = nn.Conv2d(
            in_channels=hidden_dim, out_channels=1, kernel_size=1, padding=0
        )

    def forward(self, cost_pyramid):
        """
        multi-scale cost volume interactive update to generate delta flow-map

        assume at lowest resolution, disparity flow map == 0.0, i.e. NO distinguished left and right image feature
        """
        n, _, h, w = cost_pyramid[-1].shape

        hidden_state = torch.zeros(
            size=[n, self.hidden_dim, h // 2, w // 2],
            dtype=torch.float32,
            device=cost_pyramid[0].device,
        )

        for i in np.flip(range(len(cost_pyramid))):
            cost_volume = cost_pyramid[i]

            hidden_state = self.gru_cell(cost_volume, hidden_state)

        delta_flow_map = self.out_1x1_layer(hidden_state)
        return delta_flow_map