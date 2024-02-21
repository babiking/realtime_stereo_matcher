from __future__ import print_function, division
import os
import sys
import json
import time
import glob
import logging
import onnxruntime
import numpy as np
import torch
import torch.nn.functional as F
import cv2 as cv
from model.mobile_stereo_base import MobileStereoBase
from dataset.input_padder import InputPadder
from tools.pfm_file_io import write_pfm_file
from tools.colorize import colorize_2d_matrix
import gflags

gflags.DEFINE_string(
    "base_config_json",
    "configure/stereo_base_net_v1.json",
    "experiment configure json file",
)
gflags.DEFINE_string(
    "model_chkpt_file",
    "experiments/TRAINER_BASE_V1/checkpoints/TRAINER_BASE_V1-epoch-200000.pth.gz",
    "model checkpont file",
)
gflags.DEFINE_string(
    "left",
    "/mnt/data/workspace/datasets/MyRealsense/20240220_desktop_0000/image/*_off_left_Img.png",
    "left images",
)
gflags.DEFINE_list(
    "replace",
    ["_left_Img", "_right_Img"],
    "left to right image name replace",
)
gflags.DEFINE_string(
    "output",
    "/mnt/data/workspace/datasets/MyRealsense/20240220_desktop_0000/predict",
    "output path",
)
gflags.DEFINE_boolean("use_onnx_inference", True,
                      "if set, use onnx inference instead of pytorch")

autocast = torch.cuda.amp.autocast


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def export_stereo_model_to_onnx(model, onnx_file, device):
    os.makedirs(os.path.dirname(onnx_file), exist_ok=True)

    left = torch.rand(size=(1, 3, 480, 640)) * 255.0
    left = left.to(device)
    right = torch.rand(size=(1, 3, 480, 640)) * 255.0
    right = right.to(device)

    torch.onnx.export(
        model.module,
        (left, right),
        onnx_file,
        verbose=False,
        input_names=["input0", "input1"],
        output_names=["output0"],
        opset_version=16,
    )
    return onnx_file


def run_onnx_inference(l_img, r_img, onnx_file):
    session = onnxruntime.InferenceSession(onnx_file)

    outputs = session.run(
        None,
        {
            "input0": l_img.cpu().detach().numpy().astype(np.float32),
            "input1": r_img.cpu().detach().numpy().astype(np.float32)
        },
    )
    return outputs


def main():
    FLAGS = gflags.FLAGS
    FLAGS(sys.argv)

    logging.basicConfig(
        level=logging.INFO,
        format=
        "%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    )

    base_config = json.load(open(FLAGS.base_config_json, "r"))

    model = torch.nn.DataParallel(MobileStereoBase(base_config)).to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    model.cuda()
    model.eval()

    logging.info(f"Loading checkpoint: {FLAGS.model_chkpt_file}...")
    checkpoint = torch.load(FLAGS.model_chkpt_file)
    model.load_state_dict(checkpoint, strict=True)
    logging.info(f"Done loading checkpoint.")

    print(
        f"The model has {format(count_parameters(model)/1e6, '.4f')}M learnable parameters."
    )

    model_path = os.path.dirname(FLAGS.model_chkpt_file)
    model_name = os.path.splitext(os.path.basename(FLAGS.model_chkpt_file))[0]
    onnx_file = os.path.join(model_path, f"{model_name}.onnx")
    if FLAGS.use_onnx_inference and not os.path.exists(onnx_file):
        export_stereo_model_to_onnx(
            model,
            onnx_file=onnx_file,
            device=torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"))

    # The CUDA implementations of the correlation volume prevent half-precision
    # rounding errors in the correlation lookup. This allows us to use mixed precision
    # in the entire forward pass, not just in the GRUs & feature extractors.
    use_mixed_precision = True

    save_path = os.path.join(FLAGS.output, model_name)
    os.makedirs(save_path, exist_ok=True)

    for l_img_file in glob.glob(FLAGS.left):
        l_suffix, r_suffix = FLAGS.replace

        r_img_file = l_img_file.replace(l_suffix, r_suffix)
        if not os.path.exists(r_img_file):
            continue

        l_img_name = \
            os.path.splitext(os.path.basename(l_img_file))[0].replace(l_suffix, "")

        l_img = cv.imread(l_img_file, cv.IMREAD_COLOR)
        r_img = cv.imread(r_img_file, cv.IMREAD_COLOR)

        h, w = l_img.shape[:2]

        l_img = l_img[:, :, ::-1]
        r_img = r_img[:, :, ::-1]

        l_img = torch.from_numpy(l_img.copy()).permute(2, 0, 1).float()
        r_img = torch.from_numpy(r_img.copy()).permute(2, 0, 1).float()

        l_img = l_img[None].cuda()
        r_img = r_img[None].cuda()

        with autocast(enabled=use_mixed_precision):
            start = time.time()
            if FLAGS.use_onnx_inference:
                flow_pr = run_onnx_inference(
                    l_img=F.interpolate(l_img,
                                        size=(480, 640),
                                        mode="bilinear",
                                        align_corners=True),
                    r_img=F.interpolate(r_img,
                                        size=(480, 640),
                                        mode="bilinear",
                                        align_corners=True),
                    onnx_file=onnx_file)[-1]
                flow_pr = torch.tensor(flow_pr, dtype=torch.float32).squeeze(0)
            else:
                padder = InputPadder(l_img.shape, divis_by=32)
                l_img, r_img = padder.pad(l_img, r_img)

                flow_pr = model(l_img, r_img)[-1]

                flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)
            end = time.time()

            fps = 1.0 / (end - start)
            print(f"The model inference on {l_img_file} FPS: {fps:.4f}.")

        flow_pr = -1.0 * np.squeeze(flow_pr.cpu().detach().numpy(),
                                    axis=0).astype(np.float32)
        flow_color = colorize_2d_matrix(flow_pr, min_val=1.0, max_val=64.0)

        flow_pfm_file = os.path.join(save_path,
                                     f"{l_img_name}_{w}x{h}_disparity.pfm")
        write_pfm_file(flow_pfm_file, np.flipud(flow_pr), 1.0)

        flow_color_file = os.path.join(save_path,
                                       f"{l_img_name}_{w}x{h}_disparity.png")
        cv.imwrite(flow_color_file, flow_color)


if __name__ == "__main__":
    main()
