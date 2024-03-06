import numpy as np
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


class LossFactory(nn.Module):
    def __init__(self, loss_configs, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.loss_configs = loss_configs

        self.loss_funcs = []
        self.loss_weights = []
        for loss_config in loss_configs:
            loss_type = loss_config["type"]
            if loss_type == "SequenceLoss":
                self.loss_funcs.append(SequenceLoss(**loss_config["parameters"]))
            elif loss_type == "LRConsistentLoss":
                self.loss_funcs.append(LRConsistentLoss(**loss_config["parameters"]))
            else:
                raise NotImplementedError(f"invalid loss type: {loss_type}!")

            self.loss_weights.append(loss_config.get("weight", 1.0))

    def forward(self, l_flow_gt, l_valid_gt, l_flow_preds, l_fmaps, r_fmaps):
        loss = 0.0

        for loss_weight, loss_func in zip(self.loss_weights, self.loss_funcs):
            loss += loss_weight * loss_func(
                l_flow_gt, l_valid_gt, l_flow_preds, l_fmaps, r_fmaps
            )
        return loss


class BaseLoss(nn.Module):
    def __init__(self, loss_gamma, max_flow_magnitude, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.loss_gamma = loss_gamma
        self.max_flow_magnitude = max_flow_magnitude

    def get_loss_item(self, i, n, l_flow_gt, l_flow_pred, l_fmap, r_fmap):
        raise NotImplementedError

    def forward(self, l_flow_gt, l_valid_gt, l_flow_preds, l_fmaps, r_fmaps):
        """
        calculate sequence loss between flow map groundtruth and predictions.

        Args:
            [1] l_flow_gt, left flow map groundtruth, N x 1 x H x W or N x 2 x H x W, torch.float32
            [2] l_valid_gt, left valid mask groundtruth, N x 1 x H x W, torch.uint8, 1 -> valid | 0 -> invalid
            [3] l_flow_preds, left flow map predictions, [N x 1 x H x W] or [N x 2 x H x W], list
            [4] l_fmaps, left feature maps, [N x C x H' x W'], list
            [5] r_fmaps, right feature maps, [N x C x H' x W'], list

        Return:
            [1] l_flow_loss: scalar flow map loss value.
        """
        n, c, h, w = l_flow_gt.shape

        n_preds = len(l_flow_preds)
        assert n_preds >= 1, f"empty flow predictions ({n_preds})!"

        l_flow_mag = torch.sum(l_flow_gt**2, dim=1).sqrt().unsqueeze(1)

        l_flow_valid = (l_valid_gt >= 0.5) & (l_flow_mag < self.max_flow_magnitude)
        l_flow_valid = l_flow_valid.repeat([1, c, 1, 1])

        assert l_flow_valid.shape == l_flow_gt.shape
        assert not torch.isinf(l_flow_gt[l_flow_valid.bool()]).any()

        l_flow_loss = 0.0
        for i in range(n_preds):
            l_flow_weight = self.loss_gamma ** (n_preds - 1 - i)
            l_flow_pred_i = l_flow_preds[i]

            assert not torch.isnan(l_flow_pred_i).any()
            assert not torch.isinf(l_flow_pred_i).any()

            if l_flow_gt.shape[-2:] != l_flow_pred_i.shape[-2:]:
                scale = float(l_flow_gt.shape[-1]) / l_flow_pred_i.shape[-1]
                l_flow_pred_i = F.interpolate(
                    l_flow_pred_i * scale,
                    size=(l_flow_gt.shape[-2:]),
                    mode="bilinear",
                    align_corners=False,
                )

            loss_item = self.get_loss_item(
                i, n_preds, l_flow_gt, l_flow_pred_i, l_fmaps[i], r_fmaps[i]
            )

            l_flow_loss += l_flow_weight * loss_item[l_flow_valid.bool()].mean()
        return l_flow_loss


class SequenceLoss(BaseLoss):
    def __init__(self, loss_gamma=0.9, max_flow_magnitude=700, *args, **kwargs) -> None:
        super().__init__(loss_gamma, max_flow_magnitude, *args, **kwargs)

        self.l1_loss_func = nn.L1Loss(reduction="none")
        self.smooth_l1_loss_func = nn.SmoothL1Loss(reduction="none", beta=1.0)

    def get_loss_item(self, i, n, l_flow_gt, l_flow_pred, l_fmap=None, r_fmap=None):
        if i == n - 1:
            return self.smooth_l1_loss_func(l_flow_gt, l_flow_pred)
        else:
            return self.l1_loss_func(l_flow_gt, l_flow_pred)


class LRConsistentLoss(BaseLoss):
    def __init__(self, loss_gamma=0.9, max_flow_magnitude=700, *args, **kwargs) -> None:
        super().__init__(loss_gamma, max_flow_magnitude, *args, **kwargs)

        self.loss_func = nn.L1Loss(reduction="none")

    def warp_by_flow_map(self, img, flow):
        """
        warp image according to stereo flow map (i.e. disparity map)

        Args:
            [1] img, N x C x H x W, original image or feature map
            [2] flow,  N x 1 x H x W or N x 2 x H x W, flow map

        Return:
            [1] warped, N x C x H x W, warped image or feature map
        """
        n, c, h, w = flow.shape

        assert c == 1 or c == 2, f"invalid flow map dimension 1 or 2 ({c})!"

        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=img.device, dtype=img.dtype),
            torch.arange(w, device=img.device, dtype=img.dtype),
            indexing="ij",
        )

        grid_x = grid_x.view([1, 1, h, w]) - flow[:, 0, :, :].view([n, 1, h, w])
        grid_x = grid_x.permute([0, 2, 3, 1])

        if c == 2:
            grid_y = grid_y.view([1, 1, h, w]) - flow[:, 1, :, :].view([n, 1, h, w])
            grid_y = grid_y.permute([0, 2, 3, 1])
        else:
            grid_y = grid_y.view([1, h, w, 1]).repeat(n, 1, 1, 1)

        grid_x = 2.0 * grid_x / (w - 1.0) - 1.0
        grid_y = 2.0 * grid_y / (h - 1.0) - 1.0
        grid_map = torch.concatenate((grid_x, grid_y), dim=-1)

        warped = F.grid_sample(
            img, grid_map, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        return warped

    def get_loss_item(self, i, n, l_flow_gt, l_flow_pred, l_fmap, r_fmap):
        assert l_flow_pred.shape[-2:] == l_fmap.shape[-2:]

        l_fmap_warp = self.warp_by_flow_map(r_fmap, l_flow_pred)

        return self.loss_func(l_fmap, l_fmap_warp).mean(1).unsqueeze(1)


class AdaptiveLoss(nn.Module):
    def __init__(
        self,
        use_recon_loss=True,
        use_smooth_loss=True,
        use_consist_loss=True,
        use_dual_transform=True,
        recon_alpha=0.85,
        ssim_win_size=11,
        ssim_sigma=1.5,
        ssim_c1=0.01**2,
        ssim_c2=0.03**2,
        smooth_beta=1.0,
        smooth_scale=20.0,
        recon_weight=1.0,
        smooth_weight=1.0,
        consist_weight=1.0,
        loss_gamma=0.7,
        margin_ratio=0.2,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.use_recon_loss = use_recon_loss
        self.use_smooth_loss = use_smooth_loss
        self.use_consist_loss = use_consist_loss
        self.use_dual_transform = use_dual_transform
        self.recon_alpha = recon_alpha
        self.ssim_win_size = ssim_win_size
        self.ssim_sigma = ssim_sigma
        self.ssim_c1 = ssim_c1
        self.ssim_c2 = ssim_c2
        self.smooth_beta = smooth_beta
        self.smooth_scale = smooth_scale
        self.recon_weight = recon_weight
        self.smooth_weight = smooth_weight
        self.consist_weight = consist_weight
        self.loss_gamma = loss_gamma
        self.margin_ratio = margin_ratio

    def warp_by_flow_map(self, img, flow):
        """
        warp image according to stereo flow map (i.e. disparity map)

        Args:
            [1] img, N x C x H x W, original image or feature map
            [2] flow,  N x 1 x H x W or N x 2 x H x W, flow map

        Return:
            [1] warped, N x C x H x W, warped image or feature map
        """
        n, c, h, w = flow.shape

        assert c == 1 or c == 2, f"invalid flow map dimension 1 or 2 ({c})!"

        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=img.device, dtype=img.dtype),
            torch.arange(w, device=img.device, dtype=img.dtype),
            indexing="ij",
        )

        grid_x = grid_x.view([1, 1, h, w]) - flow[:, 0, :, :].view([n, 1, h, w])
        grid_x = grid_x.permute([0, 2, 3, 1])

        if c == 2:
            grid_y = grid_y.view([1, 1, h, w]) - flow[:, 1, :, :].view([n, 1, h, w])
            grid_y = grid_y.permute([0, 2, 3, 1])
        else:
            grid_y = grid_y.view([1, h, w, 1]).repeat(n, 1, 1, 1)

        grid_x = 2.0 * grid_x / (w - 1.0) - 1.0
        grid_y = 2.0 * grid_y / (h - 1.0) - 1.0
        grid_map = torch.concatenate((grid_x, grid_y), dim=-1)

        warped = F.grid_sample(
            img, grid_map, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        return warped

    def calculate_average_SSIM(self, x, y, c1, c2):
        """
        SSIM (Structure Similarity Index Measure) implementation based on: https://github.com/IShengFang/ES3Net/tree/main
        """
        mu_x = F.avg_pool2d(x, 3, 1, 0)
        mu_y = F.avg_pool2d(y, 3, 1, 0)

        # (input, kernel, stride, padding)
        sigma_x = F.avg_pool2d(x**2, 3, 1, 0) - mu_x**2
        sigma_y = F.avg_pool2d(y**2, 3, 1, 0) - mu_y**2
        sigma_xy = F.avg_pool2d(x * y, 3, 1, 0) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        SSIM_d = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)

        SSIM = SSIM_n / SSIM_d

        return torch.clamp((1 - SSIM) / 2.0, 0.0, 1.0)

    def calculate_guassian_SSIM(self, src_img, dst_img, win_size, sigma, c1, c2):
        """
        SSIM (Structure Similarity Index Measure) implementation based on: https://github.com/Po-Hsun-Su/pytorch-ssim
        """

        def create_gaussian_window(win_size, sigma):
            win_1d = torch.Tensor(
                [
                    np.exp(-((x - win_size // 2) ** 2) / float(2 * sigma**2))
                    for x in range(win_size)
                ]
            )
            win_1d = win_1d.view([win_size, 1])

            win_2d = torch.matmul(win_1d, win_1d.t()).float()
            return win_2d

        n, c, h, w = src_img.shape

        win_2d = create_gaussian_window(win_size, sigma)
        win_2d = win_2d.reshape([1, 1, win_size, win_size]).repeat([c, 1, 1, 1])
        win_2d = win_2d.to(src_img.device)

        src_mu = F.conv2d(
            src_img.float(), win_2d, padding=(win_size - 1) // 2, groups=c
        )
        dst_mu = F.conv2d(
            dst_img.float(), win_2d, padding=(win_size - 1) // 2, groups=c
        )

        src_mu_sq = src_mu.pow(2)
        dst_mu_sq = dst_mu.pow(2)
        pair_mu_sq = src_mu * dst_mu

        src_sigma_sq = (
            F.conv2d(src_img * src_img, win_2d, padding=(win_size - 1) // 2, groups=c)
            - src_mu_sq
        )
        dst_sigma_sq = (
            F.conv2d(dst_img * dst_img, win_2d, padding=(win_size - 1) // 2, groups=c)
            - dst_mu_sq
        )
        pair_sigma_sq = (
            F.conv2d(src_img * dst_img, win_2d, padding=(win_size - 1) // 2, groups=c)
            - pair_mu_sq
        )

        ssim_map = ((2 * pair_mu_sq + c1) * (2 * pair_sigma_sq + c2)) / (
            (src_mu_sq + dst_mu_sq + c1) * (src_sigma_sq + dst_sigma_sq + c2)
        )
        return torch.clamp((1 - ssim_map) / 2.0, 0.0, 1.0)

    def get_reconstruct_loss(
        self, l_img, r_img, l_disp, margin_ratio, alpha, l_occ=None
    ):
        n, c, h, w = l_img.shape

        margin = int(margin_ratio * w)

        l_img_warp = self.warp_by_flow_map(r_img, l_disp)
        if l_occ is not None:
            l_occ_repeat = l_occ.unsqueeze(1).repeat([1, c, 1, 1])
            l_img_warp[l_occ_repeat] = l_img[l_occ_repeat]

        ssim = self.calculate_average_SSIM(
            l_img[:, :, :, margin:],
            l_img_warp[:, :, :, margin:],
            c1=self.ssim_c1,
            c2=self.ssim_c2,
        )

        l1_diff = torch.abs(l_img[:, :, :, margin:] - l_img_warp[:, :, :, margin:])
        loss = alpha * ssim.mean() + (1.0 - alpha) * l1_diff.mean()
        return loss

    def get_smooth_loss(self, img, disp, beta):
        def get_xy_gradient(data):
            dx = data[:, :, :, 1:] - data[:, :, :, :-1]
            dy = data[:, :, 1:, :] - data[:, :, :-1, :]
            return dx, dy

        img_dx, img_dy = get_xy_gradient(img)
        img_wx = torch.exp(-10.0 * torch.mean(torch.abs(img_dx), 1, keepdim=True))
        img_wy = torch.exp(-10.0 * torch.mean(torch.abs(img_dy), 1, keepdim=True))

        disp_dx, disp_dy = get_xy_gradient(disp)
        disp_dx2, _ = get_xy_gradient(disp_dx)
        _, disp_dy2 = get_xy_gradient(disp_dy)

        return (
            torch.mean(beta * img_wx[:, :, :, 1:] * torch.abs(disp_dx2))
            + torch.mean(beta * img_wy[:, :, 1:, :] * torch.abs(disp_dy2))
        ) / 2.0

    def forward(self, l_img, r_img, l_disps, r_disps, l_occ=None, r_occ=None):
        def rgb_to_gray(rgb):
            return (
                0.299 * rgb[:, 0, :, :]
                + 0.587 * rgb[:, 1, :, :]
                + 0.114 * rgb[:, 2, :, :]
            ).unsqueeze(1)

        l_img_gray = rgb_to_gray(l_img)
        r_img_gray = rgb_to_gray(r_img)

        loss = 0.0

        n_samples = len(l_disps)

        for i in range(n_samples):
            l_disp = l_disps[i]
            r_disp = r_disps[i]

            i_weight = self.loss_gamma ** (n_samples - 1 - i)

            i_loss = 0.0

            if self.use_recon_loss:
                l_recon_loss = self.get_reconstruct_loss(
                    l_img_gray,
                    r_img_gray,
                    l_disp,
                    self.margin_ratio,
                    self.recon_alpha,
                    l_occ,
                )

                r_recon_loss = (
                    self.get_reconstruct_loss(
                        r_img_gray.flip(-1),
                        l_img_gray.flip(-1),
                        r_disp.flip(-1),
                        self.margin_ratio,
                        self.recon_alpha,
                        r_occ.flip(-1),
                    )
                    if self.use_dual_transform
                    else 0.0
                )
                i_loss += self.recon_weight * (l_recon_loss + r_recon_loss)

            if self.use_smooth_loss:
                l_smooth_loss = self.get_smooth_loss(
                    l_img_gray, l_disp / self.smooth_scale, self.smooth_beta
                )

                r_smooth_loss = (
                    self.get_smooth_loss(
                        r_img_gray, r_disp / self.smooth_scale, self.smooth_beta
                    )
                    if self.use_dual_transform
                    else 0.0
                )

                i_loss += self.smooth_weight * (l_smooth_loss + r_smooth_loss)

            if self.use_consist_loss:
                l_consist_loss = self.get_reconstruct_loss(
                    l_disp, r_disp, l_disp, self.margin_ratio, self.recon_alpha
                )

                r_consist_loss = (
                    self.get_reconstruct_loss(
                        r_disp.flip(-1),
                        l_disp.flip(-1),
                        r_disp.flip(-1),
                        self.margin_ratio,
                        self.recon_alpha,
                    )
                    if self.use_dual_transform
                    else 0.0
                )

                i_loss += self.consist_weight * (l_consist_loss + r_consist_loss)

            loss += i_weight * i_loss
        return loss
