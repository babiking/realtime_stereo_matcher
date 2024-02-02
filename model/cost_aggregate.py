import torch
import torch.nn as nn
import torch.nn.functional as F


def cost_aggregate_factory(config):
    aggregate_type = config["type"]
    if aggregate_type == "conv_3d":
        return ConvCostAggregate3D(**config["arguments"])
    else:
        raise NotImplementedError(f"invalid cost aggregate type: {aggregate_type}!")


class BaseCostAggregate3D(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def build(self):
        raise NotImplementedError

    def forward(self, cost_volume):
        """
            CostAggregate3D to filter 3D cost volume by an implicit aggregation.

            Args:
                [1] cost_volume: N x Cin x D x H x W, where D denotes max-disparity-value.
            
            Return:
                [1] cost_volume: N x Cout x D x H x W
        """
        self.aggregate = self.build()
        return self.aggregate(cost_volume)


class ConvCostAggregate3D(BaseCostAggregate3D):

    def __init__(self,
                 in_dim,
                 hidden_dims,
                 out_dim=1,
                 kernel_size=3,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        self.out_dim = out_dim
        self.kernel_size = kernel_size

    def build(self):
        aggregate = nn.ModuleList([])

        for i in range(len(self.hidden_dims)):
            if i == 0:
                aggregate.append(\
                    nn.Conv3d(self.in_dim, self.hidden_dims[0], self.kernel_size, 1, 1))
                aggregate.append(nn.BatchNorm3d(self.hidden_dims[0]))
                aggregate.append(nn.ReLU())
            else:
                aggregate.append(\
                    nn.Conv3d(self.hidden_dims[i - 1], self.hidden_dims[i], self.kernel_size, 1, 1))
                aggregate.append(nn.BatchNorm3d(self.hidden_dims[i]))
                aggregate.append(nn.ReLU())
        aggregate.append(\
            nn.Conv3d(self.hidden_dims[-1], self.out_dim, self.kernel_size, 1, 1))
        return nn.Sequential(*aggregate)
