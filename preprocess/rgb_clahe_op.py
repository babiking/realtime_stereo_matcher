import torch
import torch.nn as nn
import cv2 as cv
import numpy as np


class RGBCLAHEOperator(nn.Module):
    def __init__(
        self, clip_limit=40.0, tile_grid_size=(8, 8), use_adaptive=True, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.use_adaptive = use_adaptive

        if self.use_adaptive:
            self.clahe = cv.createCLAHE(
                clipLimit=clip_limit, tileGridSize=tile_grid_size
            )

    def forward(self, img):
        n, c, h, w = img.shape

        rgb_items = []
        for i in range(n):
            rgb_i = img[i, :, :, :].cpu().detach().numpy()
            rgb_i = np.transpose(rgb_i, [1, 2, 0]).astype(np.uint8)

            hls_i = cv.cvtColor(rgb_i, cv.COLOR_RGB2HLS)
            hls_i[:, :, 1] = (
                self.clahe.apply(hls_i[:, :, 1])
                if self.use_adaptive
                else cv.equalizeHist(hls_i[:, :, 1])
            )

            rgb_i = cv.cvtColor(hls_i, cv.COLOR_HLS2RGB)
            rgb_i = np.expand_dims(rgb_i, axis=0)

            rgb_items.append(rgb_i)
        rgb_items = np.concatenate(rgb_items, axis=0)

        rgb_items = np.transpose(rgb_items, [0, 3, 1, 2])
        return torch.tensor(rgb_items, dtype=img.dtype, device=img.device)
