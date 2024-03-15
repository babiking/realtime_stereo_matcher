from __future__ import print_function, division
import os
import sys
import time
import json
import logging
import numpy as np
import copy
import torch
import onnxruntime
from tqdm import tqdm
import dataset.stereo_datasets as datasets
from dataset.input_padder import InputPadder
import gflags

gflags.DEFINE_string(
    "onnx_file",
    "others/mad_net/MADNet.onnx",
    "onnx file for inference",
)
gflags.DEFINE_list(
    "test_datasets", ["realsense_test"], "test datasets for evaluation"
)


def run_onnx_inference(l_img, r_img, onnx_file):
    session = onnxruntime.InferenceSession(onnx_file)

    outputs = session.run(
        None,
        {
            "left": l_img.cpu().detach().numpy().astype(np.float32),
            "right": r_img.cpu().detach().numpy().astype(np.float32),
        },
    )
    return outputs


@torch.no_grad()
def validate_realsense(onnx_file):
    """Peform validation using the Realsense (train) split"""
    aug_params = {}
    val_dataset = datasets.RealsenseDataset(aug_params, split="test")

    out_list, epe_list, fps_list = [], [], []
    for val_id in range(len(val_dataset)):
        _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None]
        image2 = image2[None]

        image1 = 2.0 * (image1 / 255.0) - 1.0
        image2 = 2.0 * (image2 / 255.0) - 1.0

        padder = InputPadder(image1.shape, divis_by=1)
        image1, image2 = padder.pad(image1, image2)

        start = time.time()
        flow_pr = run_onnx_inference(image1, image2, onnx_file)[-1]
        end = time.time()

        flow_pr = torch.tensor(flow_pr, dtype=torch.float32).permute([0, 3, 1, 2])
        flow_pr = padder.unpad(flow_pr.float()).cpu().squeeze(0)
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
        logging.info(
            f"Realsense {val_id+1} out of {len(val_dataset)}. EPE: {image_epe:.4f}, D1: {image_out[1]:.4f}, FPS: {image_fps:.4f}."
        )

        epe_list.append(image_epe)
        out_list.append(image_out)
        fps_list.append(image_fps)

    epe_list = np.array(epe_list)
    out_list = np.array(out_list)
    fps_list = np.array(fps_list)

    epe = np.mean(epe_list)
    bads = 100 * np.mean(out_list, axis=0)
    fps = np.mean(fps_list)

    print(
        "Validation Realsense: EPE=%.4f, bad0.5=%.4f, bad1.0=%.4f, bad3.0=%.4f, bad5.0=%.4f, FPS=%.4f"
        % (epe, bads[0], bads[1], bads[2], bads[3], fps)
    )
    return {
        "realsense-epe": epe,
        "realsense-bad0.5": bads[0],
        "realsense-bad1.0": bads[1],
        "realsense-bad3.0": bads[2],
        "realsense-bad5.0": bads[3],
        "realsense-fps": fps,
    }


def main():
    FLAGS = gflags.FLAGS
    FLAGS(sys.argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    )

    for dataset in FLAGS.test_datasets:
        if dataset == "realsense_test":
            validate_realsense(FLAGS.onnx_file)

        else:
            raise NotImplementedError(f"invalid dataset: {dataset}!")


if __name__ == "__main__":
    main()
