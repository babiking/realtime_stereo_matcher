from __future__ import print_function, division
import os
import sys
import json
import time
import glob
import logging
import numpy as np
import torch
import cv2 as cv
from model import build_model
from dataset.input_padder import InputPadder
from tools.pfm_file_io import write_pfm_file
from tools.colorize import colorize_2d_matrix
import gflags

gflags.DEFINE_string(
    "exp_config_json",
    "configure/opencv_sgbm_config.json",
    "experiment configure json file",
)
gflags.DEFINE_string(
    "model_chkpt_file",
    "experiments/BASE_DISP_NET_C/checkpoints/BASE_DISP_NET_C-epoch-100000.pth.gz",
    "model checkpont file",
)
gflags.DEFINE_string(
    "left",
    "/mnt/data/workspace/datasets/MiddleburyQBatch/image/*_left_Img.png",
    "left images",
)
gflags.DEFINE_list(
    "replace",
    ["_left_Img", "_right_Img"],
    "left to right image name replace",
)
gflags.DEFINE_string(
    "output",
    "/mnt/data/workspace/datasets/MiddleburyQBatch/disparity",
    "output path",
)


autocast = torch.cuda.amp.autocast


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    FLAGS = gflags.FLAGS
    FLAGS(sys.argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    )

    exp_config = json.load(open(FLAGS.exp_config_json, "r"))

    model = torch.nn.DataParallel(build_model(exp_config["model"])).to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    model.cuda()
    model.eval()

    if "train" in exp_config:
        logging.info(f"Loading checkpoint: {FLAGS.model_chkpt_file}...")
        checkpoint = torch.load(FLAGS.model_chkpt_file)
        model.load_state_dict(checkpoint, strict=True)
        logging.info(f"Done loading checkpoint.")

        print(
            f"The model has {format(count_parameters(model)/1e6, '.4f')}M learnable parameters."
        )

    # The CUDA implementations of the correlation volume prevent half-precision
    # rounding errors in the correlation lookup. This allows us to use mixed precision
    # in the entire forward pass, not just in the GRUs & feature extractors.
    use_mixed_precision = exp_config["model"].get("mixed_precision", True)

    save_path = os.path.join(FLAGS.output, exp_config["name"])
    os.makedirs(save_path, exist_ok=True)

    for l_img_file in glob.glob(FLAGS.left):
        l_suffix, r_suffix = FLAGS.replace

        r_img_file = l_img_file.replace(l_suffix, r_suffix)
        if not os.path.exists(r_img_file):
            continue

        l_img_name = os.path.splitext(os.path.basename(l_img_file))[0].replace(
            l_suffix, ""
        )

        l_img = cv.imread(l_img_file, cv.IMREAD_UNCHANGED)
        r_img = cv.imread(r_img_file, cv.IMREAD_UNCHANGED)

        h, w = l_img.shape[:2]

        l_img = l_img[:, :, ::-1]
        r_img = r_img[:, :, ::-1]

        l_img = torch.from_numpy(l_img.copy()).permute(2, 0, 1).float()
        r_img = torch.from_numpy(r_img.copy()).permute(2, 0, 1).float()

        l_img = l_img[None].cuda()
        r_img = r_img[None].cuda()

        padder = InputPadder(
            l_img.shape, divis_by=2 ** exp_config["model"].get("downsample_factor", 6)
        )
        l_img, r_img = padder.pad(l_img, r_img)

        with autocast(enabled=use_mixed_precision):
            start = time.time()
            flow_pr = model(l_img, r_img)[-1]
            end = time.time()

            fps = 1.0 / (end - start)
            print(f"The model inference on {l_img_file} FPS: {fps:.4f}.")

        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)

        flow_pr = -1.0 * np.squeeze(flow_pr.cpu().detach().numpy(), axis=0).astype(
            np.float32
        )
        flow_color = colorize_2d_matrix(flow_pr, min_val=1.0, max_val=64.0)

        flow_pfm_file = os.path.join(save_path, f"{l_img_name}_{w}x{h}_disparity.pfm")
        write_pfm_file(flow_pfm_file, np.flipud(flow_pr), 1.0)

        flow_color_file = os.path.join(save_path, f"{l_img_name}_{w}x{h}_disparity.png")
        cv.imwrite(flow_color_file, flow_color)


if __name__ == "__main__":
    main()
