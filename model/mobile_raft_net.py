# MobileRaftNet implementation based on https://github.com/princeton-vl/RAFT-Stereo

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, stride=1, norm_fn="group", group_dim=8):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        if norm_fn == "group":
            assert (
                out_dim % group_dim == 0
            ), f"GroupNorm with channels NOT evenly divided {out_dim} // {group_dim}!"

            num_groups = out_dim // group_dim

            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=out_dim)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_dim)
            if not (stride == 1 and in_dim == out_dim):
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=out_dim)
        elif norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(out_dim)
            self.norm2 = nn.BatchNorm2d(out_dim)
            if not (stride == 1 and in_dim == out_dim):
                self.norm3 = nn.BatchNorm2d(out_dim)
        elif norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(out_dim)
            self.norm2 = nn.InstanceNorm2d(out_dim)
            if not (stride == 1 and in_dim == out_dim):
                self.norm3 = nn.InstanceNorm2d(out_dim)
        elif norm_fn == "none":
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not (stride == 1 and in_dim == out_dim):
                self.norm3 = nn.Sequential()

        if stride == 1 and in_dim == out_dim:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=stride), self.norm3
            )

    def forward(self, x):
        y = x
        y = self.conv1(y)
        y = self.norm1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.relu(y)

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class ResNetFeatureEncoder(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dims,
        down_factor,
        norm_fn="batch",
        num_groups=8,
    ):
        super(ResNetFeatureEncoder, self).__init__()
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.down_factor = down_factor

        assert len(hidden_dims) >= down_factor + 1

        self.norm_fn = norm_fn
        self.num_groups = num_groups

        self.conv1 = nn.Conv2d(
            self.in_dim, self.hidden_dims[0], kernel_size=7, stride=1, padding=3
        )

        if self.norm_fn == "group":
            self.norm1 = nn.GroupNorm(
                num_groups=self.num_groups, num_channels=self.hidden_dims[0]
            )
        elif self.norm_fn == "batch":
            self.norm1 = nn.BatchNorm2d(self.hidden_dims[0])
        elif self.norm_fn == "instance":
            self.norm1 = nn.InstanceNorm2d(self.hidden_dims[0])
        elif self.norm_fn == "none":
            self.norm1 = nn.Sequential()

        self.relu1 = nn.ReLU(inplace=True)

        self.hidden_layers = nn.ModuleList([])
        for i in range(len(self.hidden_dims)):
            self.hidden_layers.append(
                self._make_layer(
                    in_dim=self.hidden_dims[max(i - 1, 0)],
                    out_dim=self.hidden_dims[i],
                    stride=1 if i == 0 else 2,
                )
            )

        self.out_layers = nn.ModuleList([])
        for i in range(self.down_factor, len(self.hidden_dims)):
            self.out_layers.append(
                nn.Sequential(
                    ResidualBlock(
                        self.hidden_dims[i],
                        self.hidden_dims[i],
                        stride=1,
                        norm_fn="instance",
                    ),
                    nn.Conv2d(self.hidden_dims[i], self.hidden_dims[i], 3, padding=1),
                )
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_dim, out_dim, stride=1):
        return nn.Sequential(
            ResidualBlock(in_dim, out_dim, stride=stride, norm_fn=self.norm_fn),
            ResidualBlock(out_dim, out_dim, stride=1, norm_fn=self.norm_fn),
        )

    def forward(self, x):
        # x: 1 x 3 x 480 x 640
        # conv1: 1 x 64 x 480 x 640
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        extract_outs = []
        for i, hidden_layer in enumerate(self.hidden_layers):
            x = hidden_layer(x)

            if i >= self.down_factor:
                extract_outs.append(self.out_layers[i - self.down_factor](x))
        return extract_outs


def make_correlation_pyramid(l_fmap, r_fmap, corr_levels):
    n, _, h, w1 = l_fmap.shape
    n, _, h, w2 = r_fmap.shape

    # corr_volume (4x downsample): 1 x 120 x 160 x 160
    corr_volume = torch.einsum("aijk,aijh->ajkh", l_fmap, r_fmap)
    corr_volume = corr_volume.view(n * h * w1, 1, 1, w2)

    corr_pyramid = []
    for i in range(corr_levels):
        if i > 0:
            corr_volume = F.avg_pool2d(corr_volume, kernel_size=[1, 2], stride=[1, 2])
        corr_pyramid.append(corr_volume)
    return corr_pyramid


def init_grid_map(shape, dtype, device):
    n, _, h, w = shape

    _, grid_x = torch.meshgrid(
        torch.arange(h, dtype=dtype, device=device),
        torch.arange(w, dtype=dtype, device=device),
        indexing="ij",
    )
    grid_x = grid_x.view([1, 1, h, w]).repeat([n, 1, 1, 1])
    return grid_x


def sample_correlation_pyramid(corr_pyramid, corr_radius, grid_x, flow_map):
    n, _, h, w = grid_x.shape

    x0 = grid_x + flow_map
    x0 = x0.reshape([n * h * w, 1, 1, 1])

    dx = torch.linspace(
        -corr_radius,
        corr_radius,
        2 * corr_radius + 1,
        dtype=flow_map.dtype,
        device=flow_map.device,
    )

    sample_pyramid = []
    for i, corr_volume in enumerate(corr_pyramid):
        x = (x0 / 2**i) + dx
        x = (2.0 * x) / (corr_volume.shape[-1] - 1.0) - 1.0

        y = torch.zeros_like(x)

        sample_volume = F.grid_sample(
            corr_volume,
            torch.concat([x, y], dim=-1),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        sample_volume.view(n, h, w, 2 * corr_radius + 1)

        sample_pyramid.append(sample_volume)
    sample_pyramid = torch.concat(sample_pyramid, dim=-1)
    sample_pyramid = sample_pyramid.permute([0, 3, 1, 2]).contiguous().float()

    return sample_pyramid


class ConvGRU(nn.Module):
    def __init__(self, in_dim, hidden_dim, kernel_size=3):
        super(ConvGRU, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size

        self.conv_xz = nn.Conv2d(
            in_dim, hidden_dim, kernel_size, padding=(kernel_size - 1) // 2
        )
        self.conv_hz = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size, padding=(kernel_size - 1) // 2
        )

        self.conv_xr = nn.Conv2d(
            in_dim, hidden_dim, kernel_size, padding=(kernel_size - 1) // 2
        )
        self.conv_hr = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size, padding=(kernel_size - 1) // 2
        )

        self.conv_xq = nn.Conv2d(
            in_dim, hidden_dim, kernel_size, padding=(kernel_size - 1) // 2
        )
        self.conv_hq = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size, padding=(kernel_size - 1) // 2
        )

    def forward(self, x, h, z_bias=0.0, r_bias=0.0, q_bias=0.0):
        """
        GRU cell implemented with Conv2D.

            z[t] = sigmoid(W_xz @ x[t] + W_hz @ h[t - 1])
            r[t] = sigmoid(W_xr @ x[t] + W_hr @ h[t - 1])
            h^[t] = q[t] = tanh(W_xq @ x[t] + r[t] .* (W_hq @ h[t - 1]))
            h[t] = (1 - z[t]) .* h^[t] + z[t] .* h[t - 1]
        """
        z = torch.sigmoid(self.conv_xz(x) + self.conv_hz(h) + z_bias)

        r = torch.sigmoid(self.conv_xr(x) + self.conv_hr(h) + r_bias)

        q = torch.tanh(self.conv_xq(x) + r * self.conv_hq(h) + q_bias)

        h = (1.0 - z) * q + z * h
        return h


class MotionCapture(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.convc1 = nn.Conv2d(in_dim, hidden_dim, 1, padding=0)
        self.convc2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.convf1 = nn.Conv2d(1, hidden_dim, 7, padding=3)
        self.convf2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)

        self.conv = nn.Conv2d(2 * hidden_dim, out_dim - 1, 3, padding=1)

    def forward(self, flow, corr):
        corr_fmap = F.relu(self.convc1(corr))
        corr_fmap = F.relu(self.convc2(corr_fmap))

        flow_fmap = F.relu(self.convf1(flow))
        flow_fmap = F.relu(self.convf2(flow_fmap))

        out_fmap = F.relu(self.conv(torch.cat([flow_fmap, corr_fmap], dim=1)))
        out_fmap = torch.concat((out_fmap, flow), dim=1)

        return out_fmap


class IterativeUpdateBlock(nn.Module):
    def __init__(
        self,
        updata_in_dims,
        motion_in_dim,
        motion_hidden_dim,
        motion_out_dim,
        flow_hidden_dim,
        mask_hidden_dim,
        mask_out_dim,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.updata_in_dims = updata_in_dims
        self.motion_in_dim = motion_in_dim
        self.motion_hidden_dim = motion_hidden_dim
        self.motion_out_dim = motion_out_dim

        self.flow_hidden_dim = flow_hidden_dim

        self.mask_hidden_dim = mask_hidden_dim
        self.mask_out_dim = mask_out_dim

        self.motion_head = MotionCapture(
            in_dim=motion_in_dim, hidden_dim=motion_hidden_dim, out_dim=motion_out_dim
        )

        self.flow_head = nn.Sequential(
            nn.Conv2d(updata_in_dims[0], flow_hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(flow_hidden_dim, 1, 3, padding=1),
        )

        self.mask_head = nn.Sequential(
            nn.Conv2d(updata_in_dims[0], mask_hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mask_hidden_dim, mask_out_dim, 1, padding=0),
        )

        self.gru_cells = nn.ModuleList([])
        for i in range(len(self.updata_in_dims)):
            upper_dim = updata_in_dims[i - 1] if i > 0 else self.motion_out_dim
            lower_dim = updata_in_dims[i + 1] if i < len(self.updata_in_dims) - 1 else 0

            self.gru_cells.append(
                ConvGRU(
                    in_dim=upper_dim + lower_dim,
                    hidden_dim=updata_in_dims[i],
                )
            )

    def downsample(self, fmap):
        return F.avg_pool2d(fmap, 3, stride=2, padding=1)

    def upsample(self, fmap, newsize):
        return F.interpolate(
            fmap,
            newsize,
            mode="bilinear",
            align_corners=True,
        )

    def forward(self, in_fmaps, corr_fmap, flow_map):
        assert len(in_fmaps) == len(self.updata_in_dims)

        for i in range(len(in_fmaps)):
            j = len(in_fmaps) - 1 - i

            if j == len(in_fmaps) - 1:
                x = self.downsample(in_fmaps[j - 1])
            elif j == 0:
                x = torch.concat(
                    (
                        self.motion_head(flow_map, corr_fmap),
                        self.upsample(in_fmaps[j + 1], in_fmaps[j].shape[2:]),
                    ),
                    dim=1,
                )
            else:
                x = torch.concat(
                    (
                        self.downsample(in_fmaps[j - 1]),
                        self.upsample(in_fmaps[j + 1], in_fmaps[j].shape[2:]),
                    ),
                    dim=1,
                )

            in_fmaps[j] = self.gru_cells[j](x=x, h=in_fmaps[j])

        delta_flow_map = self.flow_head(in_fmaps[0])

        upsample_mask = 0.25 * self.mask_head(in_fmaps[0])
        return in_fmaps, delta_flow_map, upsample_mask


def upsample_by_convex_combine(in_flow, up_mask, down_factor, flow_radius):
    n, _, h, w = in_flow.shape

    down_scale = 2**down_factor
    flow_range = 2 * flow_radius + 1

    flow_sqrt = int(np.sqrt(flow_range))
    assert flow_range == flow_sqrt**2

    assert up_mask.shape[1] == (down_scale**2) * flow_range

    up_mask = up_mask.view([n, flow_range, down_scale**2, h, w])
    up_mask = torch.softmax(up_mask, dim=1)

    up_flow = F.unfold(
        down_scale * in_flow,
        [flow_sqrt, flow_sqrt],
        padding=[(flow_sqrt - 1) // 2, (flow_sqrt - 1) // 2],
    )
    up_flow = up_flow.view([n, flow_range, 1, h, w])

    up_flow = torch.sum(up_mask * up_flow, dim=1)

    up_flow = up_flow.view(n, down_scale, down_scale, h, w)
    up_flow = up_flow.permute([0, 1, 3, 2, 4])
    up_flow = up_flow.view([n, 1, down_scale * h, down_scale * w])
    return up_flow


def upsample_by_interpolate(in_flow, out_shape):
    scale = float(in_flow.shape[-1]) / out_shape[-1]

    out_flow = F.interpolate(
        in_flow * scale, out_shape, mode="bilinear", align_corners=True
    )
    return out_flow


class MobileRaftNet(nn.Module):
    def __init__(
        self,
        in_dim=3,
        encode_hidden_dims=[32, 48, 64, 64, 64],
        down_factor=2,
        norm_fn="batch",
        num_groups=8,
        corr_levels=4,
        corr_radius=4,
        motion_hidden_dim=64,
        motion_out_dim=64,
        flow_hidden_dim=64,
        mask_hidden_dim=64,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.in_dim = in_dim
        self.encode_hidden_dims = encode_hidden_dims
        self.down_factor = down_factor
        self.norm_fn = norm_fn
        self.num_groups = num_groups
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius
        self.motion_hidden_dim = motion_hidden_dim
        self.motion_out_dim = motion_out_dim
        self.flow_hidden_dim = flow_hidden_dim
        self.mask_hidden_dim = mask_hidden_dim

        self.encoder = ResNetFeatureEncoder(
            in_dim=in_dim,
            hidden_dims=encode_hidden_dims,
            down_factor=down_factor,
            norm_fn=norm_fn,
            num_groups=num_groups,
        )

        self.updater = IterativeUpdateBlock(
            updata_in_dims=encode_hidden_dims[down_factor:],
            motion_in_dim=corr_levels * (2 * corr_radius + 1),
            motion_hidden_dim=motion_hidden_dim,
            motion_out_dim=motion_out_dim,
            flow_hidden_dim=flow_hidden_dim,
            mask_hidden_dim=mask_hidden_dim,
            mask_out_dim=(down_factor**4) * (2 * corr_radius + 1),
        )

    def forward(self, l_img, r_img, num_iters=16, flow_map=None):
        l_img = (2.0 * (l_img / 255.0) - 1.0).contiguous()
        r_img = (2.0 * (r_img / 255.0) - 1.0).contiguous()

        l_fmaps = self.encoder(l_img)
        r_fmaps = self.encoder(r_img)

        corr_pyramid = make_correlation_pyramid(
            l_fmaps[0], r_fmaps[0], self.corr_levels
        )

        l_fmaps = [F.tanh(x) for x in l_fmaps]

        grid_x = init_grid_map(
            shape=l_fmaps[0].shape, dtype=l_fmaps[0].dtype, device=l_fmaps[0].device
        )

        if flow_map is None:
            n, _, h, w = l_fmaps[0].shape
            flow_map = torch.zeros(
                size=[n, 1, h, w], dtype=l_fmaps[0].dtype, device=l_fmaps[0].device
            )

        flow_predictions = []
        for i in range(num_iters):
            corr_fmap = sample_correlation_pyramid(
                corr_pyramid, self.corr_radius, grid_x, flow_map
            )

            l_fmaps, delta_flow_map, upsample_mask = self.updater(
                l_fmaps, corr_fmap, flow_map
            )

            flow_map = flow_map + delta_flow_map

            flow_predictions.append(upsample_by_convex_combine(flow_map, upsample_mask))

        return flow_predictions
