from __future__ import print_function, division
import os
import sys
import time
import json
import logging
import numpy as np
import copy
import torch
from tqdm import tqdm
from model import build_model
import dataset.stereo_datasets as datasets
from dataset.input_padder import InputPadder
from loss.loss import AdaptiveLoss
from tools.profiler import get_model_capacity
import gflags

gflags.DEFINE_string(
    "exp_config_json",
    "configure/stereo_net_config_v3_finetune.json",
    "experiment configure json file",
)
gflags.DEFINE_string(
    "model_chkpt_file",
    "experiments/STEREO_NET_V3_FINETUNE/checkpoints/STEREO_NET_V3_FINETUNE-epoch-4000.pth.gz",
    "model checkpont file",
)

autocast = torch.cuda.amp.autocast


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def validate_realsense(model, mixed_prec=False):
    """Peform validation using the Realsense (train) split"""
    model.eval()
    aug_params = {}
    val_dataset = datasets.RealsenseDataset(aug_params, split="test")

    out_list, epe_list, fps_list = [], [], []
    for val_id in range(len(val_dataset)):
        _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=64)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            start = time.time()
            flow_pr = model(image1, image2)[-1]
            end = time.time()
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


@torch.no_grad()
def validate_eth3d(model, mixed_prec=False):
    """Peform validation using the ETH3D (train) split"""
    model.eval()
    aug_params = {}
    val_dataset = datasets.ETH3D(aug_params)

    out_list, epe_list, fps_list = [], [], []
    for val_id in range(len(val_dataset)):
        _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=64)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            start = time.time()
            flow_pr = model(image1, image2)[-1]
            end = time.time()
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
            f"ETH3D {val_id+1} out of {len(val_dataset)}. EPE: {image_epe:.4f}, D1: {image_out[1]:.4f}, FPS: {image_fps:.4f}."
        )

        if image_epe > 80.0:
            continue

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
        "Validation ETH3D: EPE=%.4f, bad0.5=%.4f, bad1.0=%.4f, bad3.0=%.4f, bad5.0=%.4f, FPS=%.4f"
        % (epe, bads[0], bads[1], bads[2], bads[3], fps)
    )
    return {
        "eth3d-epe": epe,
        "eth3d-bad0.5": bads[0],
        "eth3d-bad1.0": bads[1],
        "eth3d-bad3.0": bads[2],
        "eth3d-bad5.0": bads[3],
        "eth3d-fps": fps,
    }


@torch.no_grad()
def validate_kitti(model, mixed_prec=False):
    """Peform validation using the KITTI-2015 (train) split"""
    model.eval()
    aug_params = {}
    val_dataset = datasets.KITTI(aug_params, image_set="training")
    torch.backends.cudnn.benchmark = True

    out_list, epe_list, fps_list = [], [], []
    for val_id in range(len(val_dataset)):
        _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=64)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            start = time.time()
            flow_pr = model(image1, image2)[-1]
            end = time.time()

        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)

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
        if val_id < 9 or (val_id + 1) % 10 == 0:
            logging.info(
                f"KITTI {val_id+1} out of {len(val_dataset)}. EPE: {image_epe:.4f}, D1: {image_out[1]:.4f}, FPS: {image_fps:.4f}."
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
        "Validation KITTI: EPE=%.4f, bad0.5=%.4f, bad1.0=%.4f, bad3.0=%.4f, bad5.0=%.4f, FPS=%.4f"
        % (epe, bads[0], bads[1], bads[2], bads[3], fps)
    )
    return {
        "kitti-epe": epe,
        "kitti-bad0.5": bads[0],
        "kitti-bad1.0": bads[1],
        "kitti-bad3.0": bads[2],
        "kitti-bad5.0": bads[3],
        "kitti-fps": fps,
    }


@torch.no_grad()
def validate_things(model, mixed_prec=False):
    """Peform validation using the FlyingThings3D (TEST) split"""
    model.eval()
    val_dataset = datasets.SceneFlowDatasets(
        dstype="frames_finalpass", things_test=True
    )

    out_list, epe_list, fps_list = [], [], []
    for val_id in tqdm(range(len(val_dataset))):
        _, image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=64)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            start = time.time()
            flow_pr = model(image1, image2)[-1]
            end = time.time()
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)
        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt) ** 2, dim=0).sqrt()

        epe = epe.flatten()
        val = (
            (valid_gt.flatten() >= 0.5)
            & (flow_gt.abs().flatten() < 192)
            & (torch.isnan(flow_pr.flatten()) == 0)
            & (flow_pr.flatten() > 0.0)
        )

        out = epe > 1.0
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())
        fps_list.append(1.0 / (end - start))

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)
    fps_list = np.array(fps_list)

    epe = np.mean(epe_list)
    d1 = 100 * np.mean(out_list)
    fps = np.mean(fps_list)

    print("Validation THINGS: EPE %.4f, D1 %.4f, FPS: %.4f" % (epe, d1, fps))
    return {"things-epe": epe, "things-d1": d1, "things-fps": fps}


@torch.no_grad()
def validate_middlebury(model, split="F", mixed_prec=False):
    def rgb_to_gray(rgb):
        return (
            0.299 * rgb[:, 0, :, :] + 0.587 * rgb[:, 1, :, :] + 0.114 * rgb[:, 2, :, :]
        ).unsqueeze(1)

    """Peform validation using the Middlebury-V3 dataset"""
    loss = AdaptiveLoss(use_dual_transform=False)

    model.eval()
    aug_params = {}
    val_dataset = datasets.Middlebury(aug_params, split=split)

    out_list, epe_list, fps_list = [], [], []
    for val_id in range(len(val_dataset)):
        (imageL_file, _, _), image1, image2, flow_gt, valid_gt = val_dataset[val_id]

        image_name = os.path.basename(os.path.dirname(imageL_file))
        # if image_name in ["ArtL", "Jadeplant", "MotorcycleE", "Piano", "PianoL"]:
        #     continue

        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, divis_by=64)
        image1, image2 = padder.pad(image1, image2)

        with autocast(enabled=mixed_prec):
            start = time.time()
            flow_pr = model(image1, image2)[-1]
            end = time.time()
        flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)

        image1 = rgb_to_gray(padder.unpad(image1).cpu())
        image2 = rgb_to_gray(padder.unpad(image2).cpu())
        l1_diff = loss.get_reconstruct_loss(
            image1,
            image2,
            flow_pr.unsqueeze(0),
            0.1,
            1.0,
            (valid_gt < 0.5).unsqueeze(0),
        )
        ssim = loss.get_reconstruct_loss(
            image1,
            image2,
            flow_pr.unsqueeze(0),
            0.1,
            0.0,
            (valid_gt < 0.5).unsqueeze(0),
        )
        # import cv2 as cv
        # image1_warp = loss.warp_by_flow_map(image2, flow=-flow_gt.unsqueeze(0))
        # image1_warp[(valid_gt < 0.5).unsqueeze(0).unsqueeze(0)] = image1[(valid_gt < 0.5).unsqueeze(0).unsqueeze(0)]
        # image1_l1 = np.abs((image1 - image1_warp).cpu().detach().numpy().squeeze())
        # cv.imwrite(f"{image_name}_L.png",
        #            image1.cpu().detach().numpy().squeeze())
        # cv.imwrite(f"{image_name}_RW.png",
        #            image1_warp.cpu().detach().numpy().squeeze())
        # cv.imwrite(
        #     f"{image_name}_DIFF.png",
        #     cv.applyColorMap(image1_l1.astype(np.uint8), cv.COLORMAP_INFERNO))

        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        epe = torch.sum((flow_pr - flow_gt) ** 2, dim=0).sqrt()

        epe_flattened = epe.flatten()
        val = (
            (valid_gt.reshape(-1) >= 0.5)
            & (flow_gt[0].reshape(-1) > -1000)
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
        # logging.info(f"MIDDLEBURY {val_id+1} out of {len(val_dataset)}, {imageL_file}.")
        logging.info(
            f"MIDDLEBURY {val_id+1} out of {len(val_dataset)}. EPE: {image_epe:.4f}, D1: {image_out[1]:.4f}, L1={l1_diff.item():.4f}, SSIM={ssim.item():.4f}, FPS: {image_fps:.4f}."
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
        "Validation Middlebury: EPE=%.4f, bad0.5=%.4f, bad1.0=%.4f, bad3.0=%.4f, bad5.0=%.4f, FPS=%.4f"
        % (epe, bads[0], bads[1], bads[2], bads[3], fps)
    )
    return {
        "middlebury-epe": epe,
        "middlebury-bad0.5": bads[0],
        "middlebury-bad1.0": bads[1],
        "middlebury-bad3.0": bads[2],
        "middlebury-bad5.0": bads[3],
        "middlebury-fps": fps,
    }


def main():
    FLAGS = gflags.FLAGS
    FLAGS(sys.argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    )

    exp_config = json.load(open(FLAGS.exp_config_json, "r"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.DataParallel(build_model(exp_config["model"])).to(device)
    model.eval()

    sample = torch.rand(size=(1, 3, 480, 640), dtype=torch.float32).to(device)
    _ = get_model_capacity(
        module=copy.deepcopy(model.module), inputs=(sample, sample), verbose=True
    )
    del sample
    torch.cuda.empty_cache()

    if "train" in exp_config:
        logging.info(f"Loading checkpoint: {FLAGS.model_chkpt_file}...")
        checkpoint = torch.load(FLAGS.model_chkpt_file)
        try:
            model.load_state_dict(checkpoint, strict=True)
        except:
            model.load_state_dict(checkpoint["model"], strict=True)
        logging.info(f"Done loading checkpoint.")

        # print(
        #     f"The model has {format(count_parameters(model)/1e6, '.4f')}M learnable parameters."
        # )

    # The CUDA implementations of the correlation volume prevent half-precision
    # rounding errors in the correlation lookup. This allows us to use mixed precision
    # in the entire forward pass, not just in the GRUs & feature extractors.
    use_mixed_precision = exp_config["model"].get("mixed_precision", True)

    for dataset in exp_config["test"]["datasets"]:
        if dataset == "realsense_test":
            validate_realsense(model, mixed_prec=use_mixed_precision)

        elif dataset == "eth3d":
            validate_eth3d(model, mixed_prec=use_mixed_precision)

        elif dataset == "kitti":
            validate_kitti(model, mixed_prec=use_mixed_precision)

        elif dataset in ([f"middlebury_{s}" for s in "FHQ"] + ["middlebury_2014"]):
            validate_middlebury(
                model,
                split=dataset.split("_")[-1],
                mixed_prec=use_mixed_precision,
            )

        elif dataset == "things":
            validate_things(model, mixed_prec=use_mixed_precision)


if __name__ == "__main__":
    main()
