import torch
import torch.nn as nn
import torch.nn.functional as F


def get_flow_map_metrics(flow_gt, flow_pred, flow_valid):
    flow_valid = flow_valid >= 0.5
    flow_valid = flow_valid.unsqueeze(1)

    epe = torch.sum((flow_pred - flow_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[flow_valid.bool().view(-1)]

    flow_metrics = {
        "epe": epe.mean().item(),
        "0.5px": (epe < 0.5).float().mean().item(),
        "1px": (epe < 1).float().mean().item(),
        "3px": (epe < 3).float().mean().item(),
        "5px": (epe < 5).float().mean().item(),
        "min": torch.min(flow_pred[0]).float().item(),
        "max": torch.max(flow_pred[0]).float().item(),
    }
    return flow_metrics


class SequenceLoss(nn.Module):
    def __init__(self, loss_gamma=0.9, max_flow_magnitude=700, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.loss_gamma = loss_gamma
        self.max_flow_magnitude = max_flow_magnitude

        self.l1_loss_func = nn.L1Loss(reduction="none")
        self.smooth_l1_loss_func = nn.SmoothL1Loss(reduction="none", beta=1.0)

    def forward(self, flow_preds, flow_gt, flow_valid):
        """
        calculate sequence loss between flow map groundtruth and predictions.

        Args:
            [1] flow_preds: flow map predictions from low to high resolution or from initial to final iterations.
            [2] flow_gt:    flow map groundtruth.
            [3] flow_valid: flow map valid mask.
            [4] loss_gamma
            [5] max_flow_magnitude

        Return:
            [1] flow_loss: scalar flow map loss value.
        """

        n_preds = len(flow_preds)

        assert n_preds >= 1, f"empty flow predictions ({n_preds})!"

        flow_mag = torch.sum(flow_gt**2, dim=1).sqrt()

        flow_valid = (flow_valid >= 0.5) & (flow_mag < self.max_flow_magnitude)
        flow_valid = flow_valid.unsqueeze(1)

        assert flow_valid.shape == flow_gt.shape, [flow_valid.shape, flow_gt.shape]
        assert not torch.isinf(flow_gt[flow_valid.bool()]).any()

        flow_loss = 0.0
        for i in range(n_preds):
            i_weight = self.loss_gamma ** (n_preds - 1 - i)
            i_flow_pred = flow_preds[i]

            assert not torch.isnan(i_flow_pred).any()
            assert not torch.isinf(i_flow_pred).any()

            if i_flow_pred.shape != flow_gt.shape:
                i_scale = float(flow_gt.shape[-1]) / i_flow_pred.shape[-1]
                i_flow_pred = F.interpolate(i_flow_pred * i_scale, (flow_gt.shape[2:]))

            if i == n_preds - 1:
                i_loss = self.smooth_l1_loss_func(flow_gt, i_flow_pred)
            else:
                i_loss = self.l1_loss_func(flow_gt, i_flow_pred)

            assert i_loss.shape == flow_valid.shape
            flow_loss += i_weight * i_loss[flow_valid.bool()].mean()
        return flow_loss
