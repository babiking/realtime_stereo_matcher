import torch
import torch.nn as nn
import torch.nn.functional as F
from model.submodule import *


def feature_extract_factory(config):
    extract_type = config["type"]
    if extract_type == "unet":
        return UNetFeatureExtract(**config["arguments"])
    else:
        raise NotImplementedError(
            f"invalid feature extractor type: {extract_type}!")


class BaseFeatureExtract(nn.Module):

    def __init__(
        self,
        hidden_dims,
        use_pretrain,
        *args,
        **kwargs,
    ) -> None:
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
                layer = nn.Sequential(\
                    SameConv2d(self.hidden_dims[i - 1], self.hidden_dims[i], 4, 2),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(self.hidden_dims[i], self.hidden_dims[i], 3, 1, 1),
                    nn.LeakyReLU(0.2),
                )
            elif i == self.down_factor:
                layer = nn.Sequential(\
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

            layer = UpMergeConvT2d(in_dim=self.hidden_dims[j],
                                   out_dim=self.hidden_dims[j - 1],
                                   cat_dim=self.hidden_dims[j - 1])
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
