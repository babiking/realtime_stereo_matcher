from __future__ import print_function, division
import os
import sys
import time
import imagesize
import logging
import numpy as np
import torch
import dataset.stereo_datasets as datasets
from dataset.frame_utils import readPFM
import gflags

gflags.DEFINE_list("datasets", ["realsense"], "test datasets for evaluation")
gflags.DEFINE_string(
    "algorithm",
    "TRAINER_BASE_V1-epoch-200000.pth",
    "test predicts path for evaluation",
)

autocast = torch.cuda.amp.autocast


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def validate_realsense(algorithm):
    """Peform validation using the Realsense (train) split"""
    aug_params = {}
    val_dataset = datasets.RealsenseDataset(aug_params)

    out_list, epe_list, fps_list, fill_list = [], [], [], []
    for val_id in range(len(val_dataset)):
        image_files, image1, image2, flow_gt, valid_gt = val_dataset[val_id]

        image_path = os.path.dirname(image_files[0])

        image_name = os.path.splitext(os.path.basename(image_files[0]))[0]
        image_name = image_name.replace("_left_Img", "")

        w, h = imagesize.get(image_files[0])

        flow_pr_file = os.path.join(
            image_path, "../predict", algorithm, f"{image_name}_{w}x{h}_disparity.pfm"
        )
        if not os.path.exists(flow_pr_file):
            continue

        start = time.time()
        flow_pr = torch.tensor(
            readPFM(flow_pr_file)[np.newaxis, :, :].copy(), dtype=torch.float32
        )
        end = time.time()

        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt) ** 2, dim=0).sqrt()

        epe_flattened = epe.flatten()
        val = (
            (valid_gt.flatten() >= 0.5)
            & (torch.isnan(flow_pr.flatten()) == 0)
            & (flow_pr.flatten() > 0.0)
        )
        out_0_5 = epe_flattened > 0.5
        out_1_0 = epe_flattened > 1.0
        out_3_0 = epe_flattened > 3.0
        out_5_0 = epe_flattened > 5.0
        image_out = [
            out_0_5[val].float().mean().item(),
            out_1_0[val].float().mean().item(),
            out_3_0[val].float().mean().item(),
            out_5_0[val].float().mean().item(),
        ]
        image_epe = epe_flattened[val].mean().item()
        image_fps = 1.0 / (end - start)
        image_fill = val.float().mean().item()
        logging.info(
            f"Realsense {val_id+1} out of {len(val_dataset)}. EPE: {image_epe:.4f}, D1: {image_out[1]:.4f}, FPS: {image_fps:.4f}, Fill: {image_fill:.4f}."
        )

        epe_list.append(image_epe)
        out_list.append(image_out)
        fps_list.append(image_fps)
        fill_list.append(image_fill)

    epe_list = np.array(epe_list)
    out_list = np.array(out_list)
    fps_list = np.array(fps_list)
    fill_list = np.array(fill_list)

    epe = np.mean(epe_list)
    bads = 100 * np.mean(out_list, axis=0)
    fps = np.mean(fps_list)
    fill = np.mean(fill_list)

    print(
        "Validation Realsense: EPE=%.4f, bad0.5=%.4f, bad1.0=%.4f, bad3.0=%.4f, bad5.0=%.4f, FPS=%.4f, Fill=%.4f"
        % (epe, bads[0], bads[1], bads[2], bads[3], fps, fill)
    )
    return {
        "realsense-epe": epe,
        "realsense-bad0.5": bads[0],
        "realsense-bad1.0": bads[1],
        "realsense-bad3.0": bads[2],
        "realsense-bad5.0": bads[3],
        "realsense-fps": fps,
        "realsense-fill": fill,
    }


def main():
    FLAGS = gflags.FLAGS
    FLAGS(sys.argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    )

    for dataset in FLAGS.datasets:
        if dataset == "realsense":
            validate_realsense(FLAGS.algorithm)

        else:
            raise NotImplementedError


if __name__ == "__main__":
    main()
