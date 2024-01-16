import torch
import torch.nn as nn
import numpy as np
import cv2 as cv


class OpenCVSGBMModule(nn.Module):
    def __init__(
        self,
        block_size=5,
        pre_filter_cap=63,
        min_disp=0,
        num_of_disps=128,
        speckle_range=5,
        speckle_win_size=164,
        disp12_max_diff=1,
        uniqueness_ratio=15,
        mode=1,
        p1=256,
        p2=240,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.block_size = block_size
        self.pre_filter_cap = pre_filter_cap
        self.min_disp = min_disp
        self.num_of_disps = num_of_disps
        self.speckle_range = speckle_range
        self.speckle_win_size = speckle_win_size
        self.disp12_max_diff = disp12_max_diff
        self.uniqueness_ratio = uniqueness_ratio
        self.mode = mode
        self.p1 = p1
        self.p2 = p2

        self.l_sgbm = cv.StereoSGBM_create()
        self.l_sgbm.setBlockSize(block_size)
        self.l_sgbm.setPreFilterCap(pre_filter_cap)
        self.l_sgbm.setMinDisparity(min_disp)
        self.l_sgbm.setNumDisparities(num_of_disps)
        self.l_sgbm.setSpeckleRange(speckle_range)
        self.l_sgbm.setSpeckleWindowSize(speckle_win_size)
        self.l_sgbm.setDisp12MaxDiff(disp12_max_diff)
        self.l_sgbm.setUniquenessRatio(uniqueness_ratio)
        self.l_sgbm.setMode(mode)
        self.l_sgbm.setP1(p1)
        self.l_sgbm.setP2(p2)

    def forward(self, l_img, r_img):
        n, c, h, w = l_img.shape

        l_disps = []
        for i in range(n):
            l_img_i = l_img[i, :, :, :].permute([1, 2, 0]).cpu().detach().numpy()
            r_img_i = r_img[i, :, :, :].permute([1, 2, 0]).cpu().detach().numpy()

            if c == 3:
                l_img_i = cv.cvtColor(l_img_i, cv.COLOR_BGR2GRAY)
                r_img_i = cv.cvtColor(r_img_i, cv.COLOR_BGR2GRAY)

            l_disp_i = self.l_sgbm.compute(
                l_img_i.astype(np.uint8), r_img_i.astype(np.uint8)
            )
            l_disp_i = l_disp_i.astype(np.float32) / 16.0
            l_disp_i = np.clip(l_disp_i, a_min=0.0, a_max=None)
            l_disp_i *= -1.0

            l_disp_i = l_disp_i.reshape(1, 1, h, w)

            l_disps.append(l_disp_i)

        l_disps = np.concatenate(l_disps, axis=0)
        l_disps = torch.tensor(l_disps, dtype=torch.float32, device=l_img.device)
        return [l_disps]
