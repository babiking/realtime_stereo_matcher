import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.submodule import *


def feature_extract_factory(config):
    extract_type = config["type"]
    if extract_type == "unet":
        return UNetFeatureExtract(**config["arguments"])
    elif extract_type == "mobile_residual":
        return MobileResidualFeatureExtract(**config["arguments"])
    elif extract_type == "mobile_net_v2_pretrain":
        return MobileNetV2FeatureExtract(**config["arguments"])
    else:
        raise NotImplementedError(f"invalid feature extractor type: {extract_type}!")


class BaseFeatureExtract(nn.Module):
    def __init__(self, hidden_dims, use_pretrain, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.hidden_dims = hidden_dims
        self.use_pretrain = use_pretrain

    def get_down_factor(self):
        raise NotImplementedError

    def forward(self, img):
        """
        BaseFeatureExtractor to generate feature pyramid from low to high resolution.

        Args:
            [1] image: N x 3 x H x W

        Return:
            [1] feature_pyramid: e.g.
                [N x C4 x H/16 x W/16,
                 N x C3 x H/8  x W/8,
                 N x C2 x H/4  x W/4,
                 N x C1 x H/2  x W/2,
                 N x C0 x H/1  x W/1]
        """
        raise NotImplementedError


class UNetFeatureExtract(BaseFeatureExtract):
    def __init__(self, hidden_dims, use_pretrain=False, *args, **kwargs):
        super().__init__(hidden_dims, use_pretrain, *args, **kwargs)

        self.down_factor = self.get_down_factor()

        self.down_layers = nn.ModuleList([])
        self.up_layers = nn.ModuleList([])

        for i in range(self.down_factor + 1):
            if i == 0:
                layer = nn.Sequential(
                    nn.Conv2d(3, self.hidden_dims[0], 3, 1, 1),
                    nn.LeakyReLU(0.2),
                )
            elif i > 0 and i < self.down_factor:
                layer = nn.Sequential(
                    SameConv2d(self.hidden_dims[i - 1], self.hidden_dims[i], 4, 2),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(self.hidden_dims[i], self.hidden_dims[i], 3, 1, 1),
                    nn.LeakyReLU(0.2),
                )
            elif i == self.down_factor:
                layer = nn.Sequential(
                    SameConv2d(self.hidden_dims[i - 1], self.hidden_dims[i], 4, 2),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(self.hidden_dims[i], self.hidden_dims[i], 3, 1, 1),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(self.hidden_dims[i], self.hidden_dims[i], 3, 1, 1),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(self.hidden_dims[i], self.hidden_dims[i], 3, 1, 1),
                    nn.LeakyReLU(0.2),
                )
            self.down_layers.append(layer)

        for i in range(self.down_factor):
            j = self.down_factor - i

            layer = UpMergeConvT2d(
                in_dim=self.hidden_dims[j],
                out_dim=self.hidden_dims[j - 1],
                cat_dim=self.hidden_dims[j - 1],
            )
            self.up_layers.append(layer)

    def get_down_factor(self):
        return len(self.hidden_dims) - 1

    def forward(self, x):
        down_pyramid = []
        for i, down_layer in enumerate(self.down_layers):
            x = down_layer(x)
            down_pyramid.append(x)

        up_pyramid = [down_pyramid[-1]]
        for i, up_layer in enumerate(self.up_layers):
            j = self.down_factor - i

            y = up_layer(up_pyramid[i], down_pyramid[j - 1])
            up_pyramid.append(y)
        return up_pyramid


class MobileResidualFeatureExtract(BaseFeatureExtract):
    def __init__(
        self,
        hidden_dims,
        use_pretrain,
        num_layers,
        expanse_ratios=None,
        dilations=None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(hidden_dims, use_pretrain, *args, **kwargs)
        self.num_layers = num_layers
        self.expanse_ratios = (
            expanse_ratios
            if expanse_ratios is not None
            else [1 for _ in range(len(self.hidden_dims))]
        )
        self.dilations = (
            dilations
            if dilations is not None
            else [1 for _ in range(len(self.hidden_dims))]
        )
        assert len(self.hidden_dims) == len(self.expanse_ratios) == len(self.dilations)

        self.down_factor = self.get_down_factor()

        self.down_layers = nn.ModuleList([])

        for i in range(self.down_factor + 1):
            layer = self.make_layer(
                num_layers=self.num_layers[i],
                in_dim=(3 if i == 0 else self.hidden_dims[i - 1]),
                out_dim=self.hidden_dims[i],
                stride=(1 if i == 0 else 2),
                expanse_ratio=self.expanse_ratios[i],
                dilation=self.dilations[i],
            )
            self.down_layers.append(layer)

    def get_down_factor(self):
        return len(self.hidden_dims) - 1

    def make_layer(self, num_layers, in_dim, out_dim, stride, expanse_ratio, dilation):
        layers = []

        layers.append(
            MobileResidualBlockV2(
                in_dim=in_dim,
                out_dim=out_dim,
                stride=stride,
                expanse_ratio=expanse_ratio,
                dilation=dilation,
            )
        )
        layers.append(nn.LeakyReLU(0.2))

        for _ in range(num_layers):
            layers.append(
                MobileResidualBlockV2(
                    in_dim=out_dim,
                    out_dim=out_dim,
                    stride=1,
                    expanse_ratio=1,
                    dilation=1,
                )
            )
            layers.append(nn.LeakyReLU(0.2))
        return nn.Sequential(*layers)

    def forward(self, x):
        n = len(self.down_layers)

        down_pyramid = []
        for i, down_layer in enumerate(self.down_layers):
            x = down_layer(x)
            down_pyramid.append(x)
        return [down_pyramid[n - 1 - i] for i in range(n)]


class MobileNetV2FeatureExtract(BaseFeatureExtract):
    def __init__(
        self, hidden_dims=[32, 16, 24, 32], use_pretrain=True, *args, **kwargs
    ):
        super().__init__(hidden_dims, use_pretrain, *args, **kwargs)

        self.layers = [1, 2, 3, 5, 6]

        model = timm.create_model(
            "mobilenetv2_100", pretrained=self.use_pretrain, features_only=True
        )

        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1
        self.act1 = nn.LeakyReLU(0.2)  # model.act1

        self.block0 = torch.nn.Sequential(*model.blocks[0 : self.layers[0]])
        self.block1 = torch.nn.Sequential(
            *model.blocks[self.layers[0] : self.layers[1]]
        )
        self.block2 = torch.nn.Sequential(
            *model.blocks[self.layers[1] : self.layers[2]]
        )
        # self.block3 = torch.nn.Sequential(*model.blocks[self.layers[2] : self.layers[3]])
        # self.block4 = torch.nn.Sequential(*model.blocks[self.layers[3] : self.layers[4]])

    def get_down_factor(self):
        return len(self.hidden_dims) - 1

    def forward(self, x):
        x1 = self.act1(self.bn1(self.conv_stem(x)))
        x2 = self.block0(x1)
        x4 = self.block1(x2)
        x8 = self.block2(x4)
        # x16 = self.block3(x8)
        # x32 = self.block4(x16)
        return [x8, x4, x2, x1]
