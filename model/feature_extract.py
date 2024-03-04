import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.submodule import *


def feature_extract_factory(config):
    extract_type = config["type"]
    if extract_type == "unet":
        return UNetFeatureExtract(**config["arguments"])
    elif extract_type == "mobile_v2":
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


class MobileNetV2FeatureUpsample(nn.Module):
    def __init__(self, hidden_dims, up_dim, *args, **kwargs):
        super(MobileNetV2FeatureUpsample, self).__init__(*args, **kwargs)

        self.hidden_dims = hidden_dims
        self.up_dim = up_dim

        self.deconv32_16 = Conv2x(
            hidden_dims[4], hidden_dims[3], deconv=True, concat=True
        )

        self.deconv16_8 = Conv2x(
            hidden_dims[3] * 2, hidden_dims[2], deconv=True, concat=True
        )

        self.deconv8_4 = Conv2x(
            hidden_dims[2] * 2, hidden_dims[1], deconv=True, concat=True
        )
        self.conv4 = BasicConv(
            hidden_dims[1] * 2, hidden_dims[1] * 2, kernel_size=3, stride=1, padding=1
        )

        self.deconv4_2 = Conv2x(
            hidden_dims[1] * 2, hidden_dims[0], deconv=True, concat=True
        )
        self.conv2 = BasicConv(
            hidden_dims[0] * 2, hidden_dims[0] * 2, kernel_size=3, stride=1, padding=1
        )

        self.deconv2_1 = Conv2x(hidden_dims[1] * 2, up_dim, deconv=True, concat=True)
        self.conv1 = BasicConv(
            up_dim * 2, up_dim * 2, kernel_size=1, stride=1, padding=0
        )

        self.weight_init()

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

    def forward(self, down_pyramid):
        x2, x4, x8, x16, x32 = down_pyramid

        y16 = self.deconv32_16(x32, x16)
        y8 = self.deconv16_8(y16, x8)
        y4 = self.conv4(self.deconv8_4(y8, x4))
        y2 = self.conv2(self.deconv4_2(y4, x2))
        y1 = self.conv1(self.deconv2_1(y2))
        return [y8, y4, y2, y1]


class MobileNetV2FeatureExtract(BaseFeatureExtract):
    def __init__(
        self,
        layer_idxs=[1, 2, 3, 5, 6],
        hidden_dims=[16, 24, 32, 96, 160],
        use_pretrain=True,
        up_dim=16,
        *args,
        **kwargs,
    ):
        super().__init__(hidden_dims, use_pretrain, *args, **kwargs)
        self.layer_idxs = layer_idxs
        self.hidden_dims = hidden_dims
        self.use_pretrain = use_pretrain
        self.up_dim = up_dim

        model = timm.create_model(
            "mobilenetv2_100", pretrained=True, features_only=True
        )
        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1
        self.act1 = nn.ReLU()  # model.act1

        self.block0 = torch.nn.Sequential(*model.blocks[0 : layer_idxs[0]])
        self.block1 = torch.nn.Sequential(*model.blocks[layer_idxs[0] : layer_idxs[1]])
        self.block2 = torch.nn.Sequential(*model.blocks[layer_idxs[1] : layer_idxs[2]])
        self.block3 = torch.nn.Sequential(*model.blocks[layer_idxs[2] : layer_idxs[3]])
        self.block4 = torch.nn.Sequential(*model.blocks[layer_idxs[3] : layer_idxs[4]])

        self.upsample = MobileNetV2FeatureUpsample(hidden_dims, up_dim)

    def get_down_factor(self):
        return len(self.hidden_dims)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv_stem(x)))
        x2 = self.block0(x)
        x4 = self.block1(x2)
        x8 = self.block2(x4)
        x16 = self.block3(x8)
        x32 = self.block4(x16)

        return self.upsample([x2, x4, x8, x16, x32])
