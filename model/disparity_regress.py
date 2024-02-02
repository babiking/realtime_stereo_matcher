import torch
import torch.nn as nn
import torch.nn.functional as F


def disparity_regress_factory(config):
    regress_type = config["type"]
    if regress_type == "soft_argmin":
        return SoftArgminDisparityRegress()
    else:
        raise NotImplementedError(
            f"invalid disparity regress type: {regress_type}!")


class BaseDisparityRegress(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, cost_volume, max_disp):
        raise NotImplementedError


class SoftArgminDisparityRegress(BaseDisparityRegress):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, cost_volume, max_disp):
        disps = torch.arange(0,
                             max_disp,
                             device=cost_volume.device,
                             dtype=cost_volume.dtype)

        cost_volume = F.softmax(cost_volume, dim=1)
        return torch.sum(cost_volume * disps.view(1, -1, 1, 1),
                         dim=1,
                         keepdim=True)
